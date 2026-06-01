# NN-project
Project for the course "Neural Networks. Theory and Practice" in the spring semester of the academic year 2025/26.

## Project scope (important)

**Current goal: predict bounding boxes only — not animal species or other class labels.**

ENA24 COCO annotations include many species categories, but this project deliberately treats detection as a **single-class** problem: “is there an object in this region?” Models are trained and evaluated on **box localization** (IoU, precision, recall, F1, mAP). Matching predictions to ground truth does **not** require the predicted class to match the species.

| Component | How this assumption is applied |
|-----------|------------------------------|
| **Baseline** | Binary window classifier (object vs background) + sliding window + NMS |
| **YOLO** | `prepare_yolo_dataset.py` writes all boxes as class `0` (`nc: 1`, name `object`); COCO `category_id` is ignored unless you pass `--multi_class` |
| **Metrics** | Shared detection metrics in `src/detection/detection_metrics.py` — class-agnostic matching by IoU |

Species-level classification may be added later as a separate step; it is **out of scope** for the current baseline and YOLO experiments unless explicitly enabled.

## Dataset: ENA24

The project uses the **ENA24-detection** dataset for object detection in camera trap images.

Official source:
- LILA BC dataset page: [https://lila.science/datasets/ena24detection/](https://lila.science/datasets/ena24detection/)

Official files:
- Metadata: [https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json](https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json)
- Public metadata without human images: [https://storage.googleapis.com/public-datasets-lila/ena24/ena24_public.json](https://storage.googleapis.com/public-datasets-lila/ena24/ena24_public.json)
- Images archive: [https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip](https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip)

The full dataset is larger than what is needed for early development. For this reason, the project uses a small sample dataset for notebook exploration and initial experiments.

### Sample preparation

The script `scripts/prepare_ena24_sample.py` creates a small development sample in:

```text
data/ena24_sample/
    images/
    annotations.json
```

It supports two workflows:
- **Public lightweight mode**: if `--data_dir` is omitted, the script downloads the public ENA24 metadata and only the selected sample images.
- **Local mode**: if you already downloaded ENA24 manually, the script reads the local files and copies a small random subset.

Run with a local dataset directory:

```bash
python scripts/prepare_ena24_sample.py --data_dir PATH_TO_DATASET
```

Run in lightweight public mode:

```bash
python scripts/prepare_ena24_sample.py
```

### Notes

- This creates a small sample for development and testing.
- The full dataset is **not required initially**.
- If you prefer, you can download ENA24 manually once and then use `--data_dir` to prepare the sample locally.
- If your local ENA24 directory structure differs from the expected layout, update the path resolution logic in `scripts/prepare_ena24_sample.py`.

## Baseline pipeline (ResNet + sliding window)

Binary window classifier → sliding window → NMS → detection metrics (P/R/F1, mAP). See [Project scope](#project-scope-important) — no per-species labels.

**Configs**

| File | Data | Use case |
|------|------|----------|
| `src/config/baseline_config.json` | `data/ena24_sample` (20 images) | Quick dev |
| `src/config/baseline_config_full.json` | `data/ena24_full` (~8789 images) | Full baseline |

**Prerequisites**

- Activate the project venv and install dependencies (`torch`, `lightning`, `click`, etc.).
- Dataset layout: `data/<set>/images/` + `annotations.json` (COCO-style).
- Run commands from the **project root**.

**1. Train** (PyTorch Lightning; exports `checkpoints/baseline_resnet*_best.pt`)

```bash
# sample
python scripts/train_model.py --config_path src/config/baseline_config.json

# full
python scripts/train_model.py --config_path src/config/baseline_config_full.json
```

Uses GPU when CUDA is available. Training on the full set can take a long time.

**2. Evaluate** (loads best checkpoint; no training)

```bash
# sample — test split
python scripts/evaluate_baseline.py --config_path src/config/baseline_config.json --split test

# full — test split
python scripts/evaluate_baseline.py --config_path src/config/baseline_config_full.json --split test
```

Use `--split val` for the validation split. Metrics are printed in a `SUMMARY` block at the end.

**3. Legacy script** (optional: train + val eval + W&B + one visualization)

```bash
python scripts/run_baseline_model.py
```

Uses `baseline_config.json` only. Training runs when `cnn_training.train_cnn` is `true` in the config (set to `false` to evaluate an existing checkpoint only).

**Checkpoints**

- Sample: `checkpoints/baseline_resnet_best.pt`
- Full: `checkpoints/baseline_resnet_full_best.pt`

Sliding-window evaluation on the full test split (~1758 images) is slow on CPU; prefer GPU or a smaller config for smoke tests.

## YOLO

**Configs**

| File | COCO source | YOLO output | Training |
|------|-------------|-------------|----------|
| `src/config/yolo_config.json` | `data/ena24_sample` (`size: 20`) | `data/ena24_yolo` | 30 epochs, `yolov8n` |
| `src/config/yolo_config_full.json` | `data/ena24_full` | `data/ena24_yolo_full` | 50 epochs, `yolov8n` |

Shared settings: single-class (`object`), `metrics.iou_threshold: 0.5` (aligned with baseline). Checkpoints: `checkpoints/yolo_best.pt` / `yolo_full_best.pt`.

### Dataset preparation

Convert ENA24 COCO annotations to Ultralytics layout (`images/`, `labels/`, `data.yaml`). Uses the **same train/val/test split** as the baseline (`seed` + ratios).

Aligned with [Project scope](#project-scope-important): **single-class by default** (`nc: 1`, name `object`). Use `--multi_class` only if you explicitly switch to per-species detection.

```bash
# sample
python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config.json

# full ENA24
python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config_full.json

# if symlinks fail on Windows, copy images instead
python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config_full.json --copy_images
```

Re-run prepare after changing label logic so all boxes use class `0` only.

Output: `data/ena24_yolo/data.yaml` and `split_summary.json` in the output folder.

### Training

Requires [Ultralytics](https://github.com/ultralytics/ultralytics): `pip install ultralytics`

Run from the **project root** (with venv activated). Step 1 prepares labels if needed; step 2 trains and copies `best.pt` to the checkpoint path in config.

**Sample (default — quick dev)**

```bash
# 1) Prepare YOLO folders + data.yaml (skip if already done)
python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config.json

# 2) Train YOLOv8n (single-class, ~20 images from config)
python scripts/train_yolo.py --config_path src/config/yolo_config.json

# Or combine: prepare + train in one command
python scripts/train_yolo.py --config_path src/config/yolo_config.json --prepare
```

Checkpoint: `checkpoints/yolo_best.pt`  
Ultralytics logs: `runs/detect/ena24_yolo_sample/`

**Full ENA24** — switch config only (same script):

```bash
python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config_full.json
python scripts/train_yolo.py --config_path src/config/yolo_config_full.json
```

On Windows, if image symlinks fail during prepare, add `--copy_images` to both commands.

**Useful options**

```bash
# Shorter smoke train
python scripts/train_yolo.py --config_path src/config/yolo_config.json --epochs 5

# Resume an interrupted run
python scripts/train_yolo.py --config_path src/config/yolo_config.json \
  --resume runs/detect/ena24_yolo_sample/weights/last.pt
```

Edit `epochs`, `batch`, `imgsz`, or `model.weights` in `src/config/yolo_config.json` / `yolo_config_full.json` without changing the script.
