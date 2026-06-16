# Baseline vs YOLO — comparison (ENA24)

Evaluation setup shared by both runs:

## Sample subset (`ena24_sample`, smoke test)


| Setting             | Value                                                                             |
| ------------------- | --------------------------------------------------------------------------------- |
| Dataset             | `data/ena24_sample` (30 images in folder)                                         |
| Subset used         | `size: 20` (config)                                                               |
| Split               | seed `42`, 60% / 20% / 20%                                                        |
| Evaluated split     | **test** (4 images)                                                               |
| Task                | Single-class — boxes only ([Project scope](../README.md#project-scope-important)) |
| Match IoU (metrics) | 0.5                                                                               |


Configs: `src/config/baseline_config.json`, `src/config/yolo_config.json`  
Checkpoints: `checkpoints/baseline_resnet_best.pt`, `checkpoints/yolo_best.pt`  
Device: CPU

---

## Metrics (test split, 4 images)


| Metric                      | Baseline (ResNet + sliding window) | YOLOv8n |
| --------------------------- | ---------------------------------- | ------- |
| Images                      | 4                                  | 4       |
| TP                          | 0                                  | 0       |
| FP                          | **2**                              | **0**   |
| FN                          | 4                                  | 4       |
| Precision                   | 0.0000                             | 0.0000  |
| Recall                      | 0.0000                             | 0.0000  |
| F1                          | 0.0000                             | 0.0000  |
| mAP @ IoU 0.5               | 0.0000                             | 0.0000  |
| Mean IoU (best pred per GT) | 0.0000                             | 0.0000  |
| Avg detections / image      | 0.50 (after NMS)                   | 0.00    |
| Avg candidates / image      | 0.75 (before NMS)                  | —       |


---

## Per-image (test)


| Image    | GT boxes | Baseline det. (after NMS) | Baseline TP/FP/FN | YOLO det. | YOLO TP/FP/FN |
| -------- | -------- | ------------------------- | ----------------- | --------- | ------------- |
| 2505.jpg | 1        | 2                         | 0 / 2 / 1         | 0         | 0 / 0 / 1     |
| 5140.jpg | 1        | 0                         | 0 / 0 / 1         | 0         | 0 / 0 / 1     |
| 9249.jpg | 1        | 0                         | 0 / 0 / 1         | 0         | 0 / 0 / 1     |
| 5807.jpg | 1        | 0                         | 0 / 0 / 1         | 0         | 0 / 0 / 1     |


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

## Full dataset — YOLOv8n (`yolo_config_full.json`)

Checkpoint: `checkpoints/yolo_full_best.pt`  
Task: single-class detection, match IoU 0.5, `conf_threshold` 0.25, NMS IoU 0.45  
Training: 20 epochs, `yolov8n.pt`, imgsz 640, batch 16, seed 42

Two evaluation setups — **do not compare splits 1:1** (leak run used **val**, no-leak run used **test**):


| Setting         | YOLO (data leak)                                   | YOLO (no data leak)                 |
| --------------- | -------------------------------------------------- | ----------------------------------- |
| Dataset         | Full ENA24 (~8789 images)                          | Full ENA24 (~8789 images)           |
| Split strategy  | Naive random per image                             | Group-level (`split_manifest.json`) |
| Data leakage    | Yes — near-duplicates may appear in train and eval | No — whole groups held out          |
| Evaluated split | **val**                                            | **test**                            |
| Images          | 1757                                               | 1732                                |
| Evaluated at    | 2026-06-02                                         | 2026-06-12                          |


### Metrics


| Metric                      | YOLO (data leak, val) | YOLO (no data leak, test) |
| --------------------------- | --------------------- | ------------------------- |
| TP                          | 1833                  | 1629                      |
| FP                          | 38                    | 77                        |
| FN                          | 119                   | 306                       |
| Precision                   | 0.9797                | 0.9549                    |
| Recall                      | 0.9390                | 0.8419                    |
| F1                          | 0.9589                | 0.8948                    |
| mAP @ IoU 0.5               | 0.9381                | 0.8372                    |
| Mean IoU (best pred per GT) | 0.8186                | 0.7380                    |
| Avg detections / image      | 1.06                  | 0.98                      |


### Short interpretation (full YOLO)

- Metrics on the **leaky val split** are systematically higher (F1 0.96 vs 0.89, mAP 0.94 vs 0.84) — consistent with near-duplicate leakage inflating scores.
- On the **no-leak test split**, YOLO still performs strongly: precision ~0.95, recall ~0.84, mAP ~0.84 on 1732 images.
- The leaky run also shows fewer FN (119 vs 306) and FP (38 vs 77), which fits a model that has already seen very similar frames during training.

### Commands used (full YOLO, no data leak)

```bash
python scripts/train_yolo.py --config_path src/config/yolo_config_full.json

python scripts/evaluate_yolo.py --config_path src/config/yolo_config_full.json --split test \
  --output_json results/yolo_full_test.json
```

---

## Full dataset — Baseline (TODO)


| Metric      | Baseline (full, no leak)    |
| ----------- | --------------------------- |
| Config      | `baseline_config_full.json` |
| Test images | —                           |
| mAP         | —                           |
| F1          | —                           |


Fill after running `evaluate_baseline.py` with `baseline_config_full.json` on the **test** split.