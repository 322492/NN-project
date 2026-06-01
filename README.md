# NN-project
Project for the course "Neural Networks. Theory and Practice" in the spring semester of the academic year 2025/26.

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

Binary window classifier → sliding window → NMS → detection metrics (P/R/F1, mAP).

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
