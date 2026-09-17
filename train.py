from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_config
from dataset import SatMapDataset, graph_collate_fn
from model import SAMRoad

import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default="16-mixed", help="32, 16-mixed, or bf16-mixed"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--pretrain", default=None,
    help="Path to a pre-trained SAMRoad checkpoint. Compatible weights are loaded "
         "(shape-matched); mismatched layers (e.g. pair_proj after SAM-Road++ changes) "
         "are skipped and stay randomly initialized."
)
parser.add_argument(
    "--limit_train_batches", default=None, type=int,
    help="Cap the number of training mini-batches per epoch. Useful for short fine-tuning runs."
)
parser.add_argument(
    "--seed", default=None, type=int,
    help="Global random seed for reproducibility (sets torch, numpy, python random, and CUDA seeds)."
)
parser.add_argument(
    "--gpu_memory_fraction", default=None, type=float,
    help="Fraction of GPU memory to reserve (0.0–1.0). Use on shared GPUs to avoid OOM for other users."
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="sam_road",
        # track hyperparameters and run metadata
        config=config,
        # disable wandb if debugging
        mode='disabled' if dev_run else None
    )


    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Good when model architecture/input shape are fixed.
    # Disabled during fast_dev_run (disk cache) and when seeded (non-deterministic algorithm selection).
    torch.backends.cudnn.benchmark = not dev_run and args.seed is None
    torch.backends.cudnn.deterministic = args.seed is not None
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')

    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)


    net = SAMRoad(config)

    if args.pretrain:
        ckpt = torch.load(args.pretrain, map_location="cpu")
        ckpt_sd = ckpt["state_dict"]
        model_sd = dict(net.named_parameters())
        compatible = {k: v for k, v in ckpt_sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
        incompatible = [k for k in ckpt_sd if k not in compatible]
        net.load_state_dict(compatible, strict=False)
        print(f"##### Pretrain: loaded {len(compatible)} weights, skipped {len(incompatible)} #####")
        print("Skipped:", incompatible)

    train_ds, val_ds = SatMapDataset(config, is_train=True, dev_run=dev_run), SatMapDataset(config, is_train=False, dev_run=dev_run)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        persistent_workers=config.DATA_WORKER_NUM > 0,
        prefetch_factor=2 if config.DATA_WORKER_NUM > 0 else None,
        collate_fn=graph_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        persistent_workers=config.DATA_WORKER_NUM > 0,
        prefetch_factor=2 if config.DATA_WORKER_NUM > 0 else None,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger()

    # from lightning.pytorch.profilers import AdvancedProfiler
    # profiler = AdvancedProfiler(dirpath='profile', filename='result_fast_matcher')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        # strategy='ddp_find_unused_parameters_true',
        precision=args.precision,
        accumulate_grad_batches=getattr(config, 'GRAD_ACCUM_STEPS', 1),
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        # profiler=profiler
        )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)