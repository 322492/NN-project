# Animal Detection in Camera Trap Images (ENA24)

**Course:** Neural Networks. Theory and Practice — spring 2025/26  
**Team:** Kamila Korczyńska, Kamil Tasarz  
**Task:** single-class object detection (bounding boxes only, no species labels)

## Deliverables


| Artifact                           | Path                                             |
| ---------------------------------- | ------------------------------------------------ |
| **Final report** (main submission) | `[final_report.ipynb](final_report.ipynb)`       |
| **Presentation**                   | `Presentation.pptx`                              |
| **Metrics comparison**             | `[results/COMPARISON.md](results/COMPARISON.md)` |
| **Course outline**                 | `Project rules & grading - Outline.pdf`          |


Open `final_report.ipynb` for the full methodology, anti-leak split, experiments, and discussion.

## Results (summary)

Match IoU **0.5**, full ENA24 (~8789 images). Details and per-run commands: `[results/COMPARISON.md](results/COMPARISON.md)`.


| Model                              | Split                        | Eval split | F1        | mAP @ 0.5 |
| ---------------------------------- | ---------------------------- | ---------- | --------- | --------- |
| YOLOv8n                            | **No leak** (group manifest) | test       | **0.895** | **0.837** |
| YOLOv8n                            | Leaky (random per image)     | val        | 0.959     | 0.938     |
| Baseline (ResNet + sliding window) | Random per image only        | test       | 0.053     | 0.020     |


YOLO on the **no-leak test split** is the primary quantitative result. Baseline full was only evaluated on a naive random split before the anti-leak protocol was finalized; a no-leak baseline re-run was not completed (see report §10.2.1).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Run all commands from the **project root** with the venv active. GPU is used automatically when CUDA is available.

## Repository structure

```text
final_report.ipynb          # Final report (notebook)
scripts/                    # CLI: data prep, train, eval, anti-leak pipeline
src/
  config/                   # JSON configs (sample + full)
  datasets/                 # COCO I/O, splits, pHash grouping
  detection/                # IoU, NMS, metrics (shared by baseline & YOLO)
  models/baseline/          # ResNet/CNN + sliding window + Lightning
data/                       # Datasets (not in git — see below)
checkpoints/                # Best weights (not in git)
runs/                       # Ultralytics training logs (not in git)
results/                    # Eval JSON etc. (COMPARISON.md is in git)
```

## Project scope

**Goal: predict bounding boxes only — not animal species.**

ENA24 COCO annotations include many species, but models are trained as **single-class** detection (`object`). Metrics match predictions to ground truth by **IoU only** (class-agnostic). YOLO: `prepare_yolo_dataset.py` writes all boxes as class `0` unless `--multi_class` is passed.


| Component | Approach                                        |
| --------- | ----------------------------------------------- |
| Baseline  | Binary window classifier + sliding window + NMS |
| YOLO      | YOLOv8n, `nc: 1`, Ultralytics pipeline          |
| Metrics   | `src/detection/detection_metrics.py`            |


## Dataset: ENA24

Official source: [ENA24-detection on LILA](https://lila.science/datasets/ena24detection/)

Expected layout for experiments:

```text
data/ena24_full/
  images/
  annotations.json
```

**Development sample** (~20 images) for smoke tests:

```bash
python scripts/prepare_ena24_sample.py --data_dir PATH_TO_DATASET
# or lightweight public download (no local ENA24):
python scripts/prepare_ena24_sample.py
```

Output: `data/ena24_sample/images/` + `annotations.json`.

## Anti-leakage split (full dataset)

Camera trap **burst sequences** produce near-duplicate frames. A naive random split per image leaks similar frames into train and test and inflates metrics. We group images with **pHash** (Hamming ≤ 3, max group size 50), then split **whole groups** into train / val / test (`seed=42`, 60/20/20).

**Step 1 — build groups** (~15–45 min first run; hash cache on reruns):

```bash
python scripts/build_duplicate_groups.py --data_dir data/ena24_full --max_distance_threshold 3
```

**Step 2 — split groups + export manifest:**

```bash
python scripts/split_coco_by_groups.py --data_dir data/ena24_full --group_manifest data/ena24_full/metadata/group_manifest.json --seed 42 --train_ratio 0.6 --val_ratio 0.2 --export_coco_splits
```

Expect `Leakage check: True`. Writes `data/ena24_full/metadata/split_manifest.json` — used by all `*_full.json` configs (`split_strategy: "manifest"`).

**Step 3 — verify (recommended):**

```bash
python scripts/report_split_statistics.py --data_dir data/ena24_full --group_manifest data/ena24_full/metadata/group_manifest.json --split_manifest data/ena24_full/metadata/split_manifest.json --compare_random
```

**Parameter sweep** (run before fixing production defaults; threshold **3** and `max_group_size` **50** were chosen):

```bash
python scripts/sweep_phash_production_params.py --data_dir data/ena24_full --thresholds 3 4 5 6 --max_group_sizes 40 50 60 --export_previews
```

Preview collages: `data/ena24_full/metadata/cluster_previews/`.

## Baseline (ResNet + sliding window)


| Config                                 | Data           | Split    |
| -------------------------------------- | -------------- | -------- |
| `src/config/baseline_config.json`      | `ena24_sample` | random   |
| `src/config/baseline_config_full.json` | `ena24_full`   | manifest |


**Train:**

```bash
python scripts/train_model.py --config_path src/config/baseline_config.json
python scripts/train_model.py --config_path src/config/baseline_config_full.json
```

**Evaluate:**

```bash
python scripts/evaluate_baseline.py --config_path src/config/baseline_config.json --split test
python scripts/evaluate_baseline.py --config_path src/config/baseline_config_full.json --split test
```

Checkpoints: `checkpoints/baseline_resnet_best.pt`, `checkpoints/baseline_resnet_full_best.pt`.

Legacy all-in-one script (sample only): `python scripts/run_baseline_model.py`.

Full test-set sliding-window eval is slow on CPU; use GPU or the sample config for smoke tests.

## YOLO (YOLOv8n)


| Config                             | Data           | YOLO export            | Epochs (config) |
| ---------------------------------- | -------------- | ---------------------- | --------------- |
| `src/config/yolo_config.json`      | `ena24_sample` | `data/ena24_yolo`      | 30              |
| `src/config/yolo_config_full.json` | `ena24_full`   | `data/ena24_yolo_full` | **20**          |


**Prepare → train → evaluate:**

```bash
python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config.json
python scripts/train_yolo.py --config_path src/config/yolo_config.json

python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config_full.json
python scripts/train_yolo.py --config_path src/config/yolo_config_full.json

python scripts/evaluate_yolo.py --config_path src/config/yolo_config_full.json --split test \
  --output_json results/yolo_full_test.json
```

On Windows if symlinks fail: add `--copy_images` to `prepare_yolo_dataset.py`.

Checkpoints: `checkpoints/yolo_best.pt`, `checkpoints/yolo_full_best.pt`.  
Ultralytics logs: `runs/detect/ena24_yolo_sample/`, `runs/detect/ena24_yolo_full/`.

Combine prepare + train: `python scripts/train_yolo.py --config_path ... --prepare`.

## Config reference

Full configs share one split manifest:

```bash
# After anti-leak steps 1–2
python scripts/train_model.py --config_path src/config/baseline_config_full.json
python scripts/prepare_yolo_dataset.py --config_path src/config/yolo_config_full.json
python scripts/train_yolo.py --config_path src/config/yolo_config_full.json
```

Baseline, YOLO prepare, and both eval scripts read `data/ena24_full/metadata/split_manifest.json` when `split_strategy` is `manifest`.