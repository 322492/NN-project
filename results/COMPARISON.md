# Baseline vs YOLO — comparison (ENA24 sample)

Evaluation setup shared by both runs:

| Setting | Value |
|---------|--------|
| Dataset | `data/ena24_sample` (30 images in folder) |
| Subset used | `size: 20` (config) |
| Split | seed `42`, 60% / 20% / 20% |
| Evaluated split | **test** (4 images) |
| Task | Single-class — boxes only ([Project scope](../README.md#project-scope-important)) |
| Match IoU (metrics) | 0.5 |

Configs: `src/config/baseline_config.json`, `src/config/yolo_config.json`  
Checkpoints: `checkpoints/baseline_resnet_best.pt`, `checkpoints/yolo_best.pt`  
Device: CPU

---

## Metrics (test split, 4 images)

| Metric | Baseline (ResNet + sliding window) | YOLOv8n |
|--------|-----------------------------------|---------|
| Images | 4 | 4 |
| TP | 0 | 0 |
| FP | **2** | **0** |
| FN | 4 | 4 |
| Precision | 0.0000 | 0.0000 |
| Recall | 0.0000 | 0.0000 |
| F1 | 0.0000 | 0.0000 |
| mAP @ IoU 0.5 | 0.0000 | 0.0000 |
| Mean IoU (best pred per GT) | 0.0000 | 0.0000 |
| Avg detections / image | 0.50 (after NMS) | 0.00 |
| Avg candidates / image | 0.75 (before NMS) | — |

---

## Per-image (test)

| Image | GT boxes | Baseline det. (after NMS) | Baseline TP/FP/FN | YOLO det. | YOLO TP/FP/FN |
|-------|----------|---------------------------|-------------------|-----------|---------------|
| 2505.jpg | 1 | 2 | 0 / 2 / 1 | 0 | 0 / 0 / 1 |
| 5140.jpg | 1 | 0 | 0 / 0 / 1 | 0 | 0 / 0 / 1 |
| 9249.jpg | 1 | 0 | 0 / 0 / 1 | 0 | 0 / 0 / 1 |
| 5807.jpg | 1 | 0 | 0 / 0 / 1 | 0 | 0 / 0 / 1 |

---

## Short interpretation

- **Neither model** reached a true positive on this tiny test set (4 images, 4 animals total).
- **Baseline** produced more predictions (sliding window + low threshold): 2 false positives on `2505.jpg`, hence higher FP count.
- **YOLO** was more conservative (`conf_threshold: 0.25`): no predictions on test → 0 FP, but still 4 FN (missed all GT).
- Metrics are **not meaningful for ranking** on N=4; use this table as a pipeline smoke check. Re-run on a larger subset or full data before drawing conclusions.

---

## Commands used

```bash
python scripts/evaluate_baseline.py --config_path src/config/baseline_config.json --split test

python scripts/evaluate_yolo.py --config_path src/config/yolo_config.json --split test
```

## Full-data runs (TODO)

| Metric | Baseline (full) | YOLO (full) |
|--------|-----------------|-------------|
| Config | `baseline_config_full.json` | `yolo_config_full.json` |
| Test images | — | — |
| mAP | — | — |
| F1 | — | — |

Fill after running the same commands with full configs.
