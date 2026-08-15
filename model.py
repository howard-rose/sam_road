import torch
import torch.nn.functional as F
from torch import nn

# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy
import numpy as np

from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve

import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d

import wandb
import pprint
import torchvision

# Only needed for the ablation experiment of using a ViT-B model without SA-1B pre-training.
# It depends on detectron2 library. Not super important.
# import vitdet


# --- SAM-Road++ helpers: node-guided resampling and extended-line strategy ---

def find_highest_mask_point(x, y, mask, device='cuda'):
    """Snap (x, y) to the highest-scoring pixel within radius=2 in the mask."""
    H, W, D = mask.shape
    x = torch.clamp(x, 0, W)
    y = torch.clamp(y, 0, D)
    x = int(x)
    y = int(y)
    radius = torch.tensor(2)
    x_min = max(0, x - radius)
    x_max = min(W, x + radius)
    y_min = max(0, y - radius)
    y_max = min(D, y + radius)

    mask_region = mask[:, x_min:x_max, y_min:y_max].to(device)
    x_coords = torch.arange(x_min, x_max, device=device).view(-1, 1).expand(x_max - x_min, y_max - y_min)
    y_coords = torch.arange(y_min, y_max, device=device).view(1, -1).expand(x_max - x_min, y_max - y_min)
    distances = torch.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
    within_radius = (distances <= radius).to(device)
    mask_scores = mask_region[1] * within_radius + mask_region[0] * within_radius

    if mask_scores.numel() > 0:
        mask_max = torch.max(mask_scores)
        max_pos = torch.nonzero(mask_scores == mask_max)
        if len(max_pos) > 0:
            x_final = max_pos[0][0] + x_min
            y_final = max_pos[0][1] + y_min
        else:
            x_final, y_final = x, y
    else:
        x_final, y_final = x, y
    return x_final, y_final


def extract_point(x1, y1, x2, y2, image, num_points):
    """Uniformly sample num_points between two endpoints; returns 3× positions."""
    H, W = image.shape[-2:]
    x_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)
    y_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)

    x_interp = x1.unsqueeze(-1) + (x2 - x1).unsqueeze(-1) * x_values
    y_interp = y1.unsqueeze(-1) + (y2 - y1).unsqueeze(-1) * y_values

    x_interp = torch.clamp(x_interp.long(), min=0, max=W - 1)
    y_interp = torch.clamp(y_interp.long(), min=0, max=H - 1)

    x_plus_1 = torch.clamp(x_interp + 1, max=W - 1)
    y_plus_1 = torch.clamp(y_interp + 1, max=H - 1)

    x_final = torch.cat([x_interp, x_interp, x_plus_1], dim=-1)
    y_final = torch.cat([y_interp, y_plus_1, y_interp], dim=-1)
    return x_final, y_final


def extendline(points1, points2, image):
    """
    Extended-line strategy: sample road-mask values along the line between two
    points plus 8-pixel extensions beyond each endpoint.
    Returns features of shape [B, N, 150].
    """
    B, N, _ = points1.shape
    H, W = image.shape[-2:]
    extend_length = 8

    directions = (points2 - points1).float()
    lengths = torch.norm(directions, dim=2, keepdim=True)
    lengths = lengths.masked_fill(lengths == 0, 1e-8)
    directions_norm = directions / lengths

    extended_A = torch.round(points1 - directions_norm * extend_length).long()
    extended_B = torch.round(points2 + directions_norm * extend_length).long()

    # clamp — note: [:, 0] is a known quirk from the original code but extract_point re-clamps
    extended_A[:, 0] = extended_A[:, 0].clamp(0, W - 1)
    extended_A[:, 1] = extended_A[:, 1].clamp(0, H - 1)
    extended_B[:, 0] = extended_B[:, 0].clamp(0, W - 1)
    extended_B[:, 1] = extended_B[:, 1].clamp(0, H - 1)

    extend_x1, extend_y1 = extended_A[..., 0], extended_A[..., 1]
    extend_x2, extend_y2 = extended_B[..., 0], extended_B[..., 1]
    x1, y1 = points1[..., 0], points1[..., 1]
    x2, y2 = points2[..., 0], points2[..., 1]

    xf1, yf1 = extract_point(extend_x1, extend_y1, x1, y1, image, num_points=15)
    xf,  yf  = extract_point(x1, y1, x2, y2, image, num_points=20)
    xf2, yf2 = extract_point(extend_x2, extend_y2, x2, y2, image, num_points=15)

    b_idx = np.arange(B)[:, None, None]
    features1 = image[b_idx, xf1, yf1]
    features  = image[b_idx, xf,  yf]
    features2 = image[b_idx, xf2, yf2]
    return torch.cat([features1, features, features2], dim=2)  # [B, N, 150]


# --- End SAM-Road++ helpers ---


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points, mask_scores=None):
        """
        Args:
            feature_maps: [B, D, H, W]
            sample_points: [B, N_points, 2] in pixel coords (x, y)
            mask_scores: [B, 2, H, W] — if provided, snaps each point to the
                         highest-scoring pixel within radius 2 (node-guided resampling).
        Returns:
            If mask_scores is None: sampled features [B, N_points, D]
            If mask_scores provided: (snapped_features, snapped_points, original_features)
        """
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape
        device = feature_maps.device

        if mask_scores is not None:
            # Node-guided resampling: snap each GT node to highest-score pixel
            snapped_points = torch.zeros_like(sample_points)
            for bi in range(B):
                for pi in range(N_points):
                    x, y = sample_points[bi, pi]
                    if (x.item(), y.item()) == (0, 0):
                        snapped_points[bi, pi] = torch.tensor([x, y])
                    else:
                        current_mask = mask_scores[bi]  # [2, H, W]
                        x_new, y_new = find_highest_mask_point(x, y, current_mask, device)
                        snapped_points[bi, pi] = torch.tensor(
                            [x_new, y_new], dtype=torch.float32, device=device)

            norm_snapped = (snapped_points / self.config.PATCH_SIZE) * 2.0 - 1.0
            norm_snapped = norm_snapped.unsqueeze(2)
            snapped_feat = F.grid_sample(feature_maps, norm_snapped, mode='bilinear', align_corners=False)
            snapped_feat = snapped_feat.squeeze(dim=-1).permute(0, 2, 1)

            norm_orig = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
            norm_orig = norm_orig.unsqueeze(2)
            orig_feat = F.grid_sample(feature_maps, norm_orig, mode='bilinear', align_corners=False)
            orig_feat = orig_feat.squeeze(dim=-1).permute(0, 2, 1)

            return snapped_feat, snapped_points, orig_feat
        else:
            # Simple bilinear sampling (inference path)
            pts = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
            pts = pts.unsqueeze(2)
            sampled = F.grid_sample(feature_maps, pts, mode='bilinear', align_corners=False)
            return sampled.squeeze(dim=-1).permute(0, 2, 1)
    

class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        if getattr(config, 'USE_EXTENDED_LINE', False):
            # SAM-Road++: [src(128) + tgt(128) + line(150) + offset(2)] = 408
            self.pair_proj = nn.Linear(2 * self.hidden_dim + 152, self.hidden_dim)
        else:
            # Baseline: [src(128) + tgt(128) + offset(2)] = 258
            self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim)

        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, graph_points, point_features_o, pairs, pairs_valid, mask_scores=None):
        # points: [B, N_points, 2]           — tgt positions (snapped or same as graph_points)
        # point_features: [B, N_points, D]   — tgt features
        # graph_points: [B, N_points, 2]     — src (original) positions
        # point_features_o: [B, N_points, D] — src (original) features
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        # mask_scores: [B, 2, H, W] or None — required only when USE_EXTENDED_LINE=True

        use_extline = getattr(self.config, 'USE_EXTENDED_LINE', False)

        point_features = F.relu(self.feature_proj(point_features))
        point_features_o = F.relu(self.feature_proj(point_features_o))

        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)

        src_features = point_features_o[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        src_points = graph_points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        offset = tgt_points - src_points

        if use_extline and mask_scores is not None:
            # SAM-Road++: append 150-dim road-mask line features
            mask_road_dim = mask_scores[:, 1, :, :]  # [B, H, W]
            line_features = extendline(src_points, tgt_points, mask_road_dim)
            # [B, N_samples*N_pairs, 128+128+150+2 = 408]
            pair_features = torch.concat([src_features, tgt_features, line_features, offset], dim=2)
        else:
            # Baseline SAM-Road: original TOPONET_VERSION ablations still work here
            if self.config.TOPONET_VERSION == 'no_tgt_features':
                pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset], dim=2)
            if self.config.TOPONET_VERSION == 'no_offset':
                pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset)], dim=2)
            else:
                # [B, N_samples*N_pairs, 128+128+2 = 258]
                pair_features = torch.concat([src_features, tgt_features, offset], dim=2)

        pair_features = F.relu(self.pair_proj(pair_features))

        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)
        padding_mask = ~pairs_valid

        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)

        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        logits = self.output_proj(pair_features)
        scores = torch.sigmoid(logits)
        return logits, scores



class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        # self.qkv = qkv
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        # qkv = self.qkv(x)  # B,N,N,3*org_C
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv



class SAMRoad(pl.LightningModule):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h'}
        if config.SAM_VERSION == 'vit_b':
            ### SAM config (B)
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
            ###
        elif config.SAM_VERSION == 'vit_l':
            ### SAM config (L)
            encoder_embed_dim=1024
            encoder_depth=24
            encoder_num_heads=16
            encoder_global_attn_indexes=[5, 11, 17, 23]
            ###
        elif config.SAM_VERSION == 'vit_h':
            ### SAM config (H)
            encoder_embed_dim=1280
            encoder_depth=32
            encoder_num_heads=16
            encoder_global_attn_indexes=[7, 15, 23, 31]
            ###
            
        prompt_embed_dim = 256
        # SAM default is 1024
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        encoder_output_dim = prompt_embed_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        if self.config.NO_SAM:
            # Only needed for the ablation experiment of using a ViT-B model without SA-1B pre-training.
            # It depends on detectron2 library. Not super important. 
            ### im1k + mae pre-trained vitb
            # self.image_encoder = vitdet.VITBEncoder(image_size=image_size, output_feature_dim=prompt_embed_dim)
            # self.matched_param_names = self.image_encoder.matched_param_names
            raise NotImplementedError((
                "This ablation experiment depends on detectron2, "
                "which is a bit messy and is not super important, "
                "so not including in the release. "
                "If you are interested, feel free to uncomment."))
        else:
            ### SAM vitb
            self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
            )

        if self.config.USE_SAM_DECODER:
            # SAM DECODER
            # Not used, just produce null embeddings
            self.prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            self.mask_decoder=MaskDecoder(
                num_multimask_outputs=2, # keypoint, road
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        else:
            #### Naive decoder
            activation = nn.GELU
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                activation(),
                nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
            )

        
        #### TOPONet
        self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNet(config, encoder_output_dim)


        #### LORA
        if config.ENCODER_LORA:
            r = self.config.LORA_RANK
            lora_layer_selection = None
            assert r > 0
            if lora_layer_selection:
                self.lora_layer_selection = lora_layer_selection
            else:
                self.lora_layer_selection = list(
                    range(len(self.image_encoder.blocks)))  # Only apply lora to the image encoder by default
            # create for storage, then we can init them or load weights
            self.w_As = []  # These are linear layers
            self.w_Bs = []

            # lets freeze first
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            # Here, we do the surgery
            for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                # If we only want few lora layer instead of all
                if t_layer_i not in self.lora_layer_selection:
                    continue
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            # Init LoRA params
            for w_A in self.w_As:
                nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)

        #### Losses
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        if self.config.NO_SAM:
            return
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)

            ## Resize pos embeddings, if needed
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict
            
            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)

    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

    
    def forward(self, rgb, graph_points, pairs, valid):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        # When the encoder is frozen no gradients flow through it, so skip storing
        # intermediate activations entirely — saves VRAM and skips encoder backward.
        if getattr(self.config, 'FREEZE_ENCODER', False):
            with torch.no_grad():
                image_embeddings = self.image_encoder(x)
        else:
            image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)

        ## Predicts local topology
        # Simple bilinear sampling — node-guided snapping omitted for training speed
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        # Pass mask_scores only when USE_EXTENDED_LINE is active; otherwise None keeps baseline behavior
        mask_for_topo = mask_scores if getattr(self.config, 'USE_EXTENDED_LINE', False) else None
        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(
            graph_points, point_features, graph_points, point_features, pairs, valid, mask_for_topo)

        # [B, H, W, 2]
        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_logits, mask_scores, topo_logits, topo_scores
    
    def infer_masks_and_img_features(self, rgb):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        if self.config.USE_SAM_DECODER:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )
            mask_logits = F.interpolate(
                low_res_logits,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_scores = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.map_decoder(image_embeddings)
            mask_scores = torch.sigmoid(mask_logits)
        
        # [B, H, W, 2]
        mask_scores = mask_scores.permute(0, 2, 3, 1)
        return mask_scores, image_embeddings
    

    def infer_toponet(self, image_embeddings, graph_points, pairs, valid, mask_scores=None):
        # image_embeddings: [B, D, h, w]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]
        # mask_scores: [B, 2, H, W] — only needed when USE_EXTENDED_LINE=True

        # Simple bilinear sampling (no node-guided snapping at inference either)
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        topo_logits, topo_scores = self.topo_net(
            graph_points, point_features, graph_points, point_features, pairs, valid, mask_scores)
        return topo_scores


    def training_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # [B, H, W, 2]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))

        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()

        loss = mask_loss + topo_loss
        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)


        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)
        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
        topo_loss *= topo_loss_mask.unsqueeze(-1)
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()
        loss = mask_loss + topo_loss
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = mask_scores[:max_viz_num, :, :, 0]
            viz_pred_road = mask_scores[:max_viz_num, :, :, 1]
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]
            
            columns = ['rgb', 'gt_keypoint', 'gt_road', 'pred_keypoint', 'pred_road']
            data = [[wandb.Image(x.cpu().numpy()) for x in row] for row in list(zip(viz_rgb, viz_gt_keypoint, viz_gt_road, viz_pred_keypoint, viz_pred_road))]
            self.logger.log_table(key='viz_table', columns=columns, data=data)

        self.keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.road_iou.update(mask_scores[..., 1], road_mask)
        
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))
        

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        topo_f1 = self.topo_f1.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.log("topo_f1", topo_f1)
        self.keypoint_iou.reset()
        self.road_iou.reset()
        self.topo_f1.reset()

    def test_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)

        self.keypoint_pr_curve.update(mask_scores[..., 0], keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))
        
        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_pr_curve.update(topo_scores, topo_gt.unsqueeze(-1).to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f'======= {category} ======')   
            precision, recall, thresholds = pr_curve_metric.compute()
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_threshold_index = torch.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[best_threshold_index]
            print(f'Best threshold {best_threshold}, P={best_precision} R={best_recall} F1={best_f1}')
        
        print('======= Finding best thresholds ======')
        find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        find_best_threshold(self.road_pr_curve, 'road')
        find_best_threshold(self.topo_pr_curve, 'topo')


    def configure_optimizers(self):
        param_dicts = []

        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'image_encoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            }
            param_dicts.append(encoder_params)
        if self.config.ENCODER_LORA:
            # LoRA params only
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)
        
        if self.config.USE_SAM_DECODER:
            matched_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k in self.matched_param_names],
                'lr': self.config.BASE_LR * 0.1
            }
            fresh_decoder_params = {
                'params': [p for k, p in self.mask_decoder.named_parameters() if 'mask_decoder.'+k not in self.matched_param_names],
                'lr': self.config.BASE_LR
            }
            decoder_params = [matched_decoder_params, fresh_decoder_params]
        else:
            decoder_params = [{
                'params': [p for p in self.map_decoder.parameters()],
                'lr': self.config.BASE_LR
            }]
        param_dicts += decoder_params

        topo_net_params = [{
            'params': [p for p in self.topo_net.parameters()],
            'lr': self.config.BASE_LR
        }]
        param_dicts += topo_net_params
        
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')

        # optimizer = torch.optim.AdamW(param_dicts, lr=self.config.BASE_LR, betas=(0.9, 0.999), weight_decay=0.1)
        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}

