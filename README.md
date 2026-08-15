# SAM-Road++ Replication Study

Replication study of [SAM-Road++](https://github.com/earth-insights/samroadplus), built on the [SAM-Road](https://github.com/htcr/sam_road) codebase (CVPRW 2024). Evaluates the "node-guided resampling" and "extended-line" strategies from SAM-Road++ on the SpaceNet, City-Scale, and Metro Manila datasets.

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3050 4GB)
- **Driver**: NVIDIA driver supporting CUDA 12.1 or newer — [nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/)
- **Pixi**: [pixi.prefix.dev](https://pixi.prefix.dev/latest/installation/) (cross-platform package manager)
- **Go**: Required only for APLS evaluation metric — [go.dev/dl](https://go.dev/dl/)

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd sam_road
git submodule update --init --recursive
```

### 2. Install the Python environment and Go dependencies

Install [pixi](https://pixi.prefix.dev/latest/installation/) for your platform, then run:

```bash
pixi install
pixi run setup
```

`pixi install` installs all Python dependencies (PyTorch + CUDA, SAM, rtree, igraph, etc.) into an isolated environment. `pixi run setup` fetches the Go module dependencies needed for APLS evaluation. All subsequent Python commands should be prefixed with `pixi run`.

### 3. Download the SAM ViT-B checkpoint

Create the `sam_ckpts/` directory and download the base checkpoint (~375 MB):

```bash
mkdir sam_ckpts
```
```bash
# Windows
curl.exe -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -o "sam_ckpts/sam_vit_b_01ec64.pth"
```
```bash
# Linux/macOS
curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -o "sam_ckpts/sam_vit_b_01ec64.pth"
```

### 4. Download pre-trained SAM-Road checkpoints (optional, for inference)

Available at [huggingface.co/congrui/sam_road](https://huggingface.co/congrui/sam_road). Place `.ckpt` files in the project root:

- `spacenet_vitb_256_e10.ckpt` — SpaceNet checkpoint
- `cityscale_vitb_512_e10.ckpt` — City-Scale checkpoint

## Data Preparation

### SpaceNet

Download from [Google Drive](https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W) and extract to match this structure:

```
sam_road/
  spacenet/
    RGB_1.0_meter/
      AOI_2_Vegas_210__rgb.png
      AOI_2_Vegas_210__gt_graph.p
      ...
    data_split.json
```

Then generate ground-truth label masks (required for training only):

```bash
pixi run python spacenet/generate_labels.py
```

### City-Scale

Download from [Tsinghua Cloud Drive](https://cloud.tsinghua.edu.cn/d/d32cb7d4b19046ed9a42/files/?p=%2Fdata.zip) and place the `20cities/` folder under `cityscale/`:

```
sam_road/
  cityscale/
    20cities/
      region_0_sat.png
      region_0_refine_gt_graph.p
      ...
```

Then generate ground-truth label masks (required for training only):

```bash
pixi run python cityscale/generate_labels.py
```

## Running Inference

```bash
# SpaceNet
pixi run python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=spacenet_vitb_256_e10.ckpt

# City-Scale
pixi run python inferencer.py --config=config/toponet_vitb_512_cityscale.yaml --checkpoint=cityscale_vitb_512_e10.ckpt
```

Results are saved to `save/infer_<timestamp>/` with subdirectories:
- `graph/` — predicted road graphs (used for evaluation)
- `mask/` — predicted road and keypoint masks
- `viz/` — visualization overlays

## Running Training

Ground-truth label masks must be generated before training (one-time, per dataset):

```bash
pixi run generate-labels-spacenet    # SpaceNet
pixi run generate-labels-cityscale   # City-Scale
```

Then start training (16-mixed precision is the default):

```bash
pixi run train-spacenet    # SpaceNet (256x256 patches, lower VRAM requirement)
pixi run train-cityscale   # City-Scale (512x512 patches)
```

> **Low VRAM note (<=6 GB):** 16-mixed precision is already the default. Reduce `BATCH_SIZE` in the config file if you hit OOM errors. SpaceNet with 256x256 patches is significantly more memory-efficient than City-Scale.

Checkpoints are saved to `lightning_logs/`.

## Evaluation

Requires Go to be installed for the APLS metric.

Pass the output directory name (found under `save/`) to the evaluation script:

```bash
# SpaceNet
pixi run python spacenet_metrics/run_eval.py --dirs <output_dir_name>

# City-Scale
pixi run python cityscale_metrics/run_eval.py --dirs <output_dir_name>
```

You can evaluate multiple runs at once by listing additional names after `--dirs`. Use `--apls-only` or `--topo-only` to run a single metric.

Results are written to `save/<output_dir_name>/results/apls/` and `save/<output_dir_name>/results/topo/`.

## Acknowledgement

Extending appreciation to the authors of the following codebases which this project is based on:
- SAM from https://github.com/facebookresearch/segment-anything
- SAM-Road from https://github.com/htcr/sam_road
- SAM-Road++ from https://github.com/earth-insights/samroadplus
- RNGDet++ from https://github.com/TonyXuQAQ/RNGDetPlusPlus
- Road Width Detector Pipeline from https://github.com/Xiao-Alatus/road-width-detector-pipeline
