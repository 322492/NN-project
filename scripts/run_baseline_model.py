from pathlib import Path
import torch
from torch import nn

from src.detection.IoU import max_iou_with_true_boxes, match_true_boxes_with_predictions, mean_iou
from src.models.baseline.cnn_classifier import SimpleCNN
from src.models.baseline.sliding_window_detection import SlidingWindow
from scripts.train_baseline_cnn import train, prepare_data_splits
from src.detection.bbox_visualization import drew_bbox_and_save, test_image_visualize
from src.config.load_config import load_config
from src.detection.nms import non_max_suppression
from src.utils.wandb_utils import init_wandb, wandb_log, finish_wandb
from src.models.baseline.resnet_classifier import ResNetBinaryClassifier
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

config = load_config("../src/config/baseline_config.json")

#TRAIN_CNN = config["cnn_training"]["train_cnn"]
TRAIN_CNN = False
USE_WANDB_CNN = False
USE_WANDB_PIPELINE = False

checkpoint_path = Path(config["cnn_training"]["checkpoint_path"])


full_dataset, train_samples, val_samples, test_samples = prepare_data_splits(
    data_dir=config["data"]["data_dir"],
    train_ratio=config["data"]["train_ratio"],
    val_ratio=config["data"]["val_ratio"],
    seed=config["seed"]
)

print("full images:", len(full_dataset))
print("train images:", len(train_samples))
print("val images:", len(val_samples))
print("test images:", len(test_samples))

#cnn = SimpleCNN().to(device)
cnn = ResNetBinaryClassifier(pretrained=True).to(device)
criterion = nn.BCEWithLogitsLoss()

cnn_run = None
if TRAIN_CNN or not checkpoint_path.exists():
    cnn_run = init_wandb(
        config=config,
        enabled=USE_WANDB_CNN,
        project="ena24-cnn",
        job_type="cnn_training"
    )

    train(
        cnn=cnn,
        criterion=criterion,
        device=device,
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        run=cnn_run
    )
    finish_wandb(cnn_run)

else:
    print(f"Loading checkpoint: {checkpoint_path}")
    cnn.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

## sliding window
# pipeline_run = init_wandb(
#     config=config,
#     enabled=USE_WANDB_PIPELINE,
#     project="ena24-baseline",
#     job_type="pipeline_eval"
# )

cnn.eval()
detector = SlidingWindow(
    cnn=cnn,
    window_sizes=tuple(config["sliding_window"]["window_sizes"]),
    overlap_ratio=config["sliding_window"]["overlap_ratio"],
    threshold=config["sliding_window"]["threshold"],
    crop_size=config["cnn_dataset"]["crop_size"],
    device=device,
)

# ## test
# total_true_boxes = 0
# sum_iou = 0.0
#
# total_boxes_before_nms = 0
# total_boxes_after_nms = 0
#
# for idx, sample in enumerate(val_samples):
#     image_path = sample["image_path"]
#     true_bboxes = sample["bboxes"]
#
#     # sliding window
#     boxes, confidence_scores = detector.predict_window(image_path)
#     boxes_before_nms = len(boxes)
#
#     # NMS
#     boxes, confidence_scores = non_max_suppression(
#         boxes,
#         confidence_scores,
#         iou_threshold=config["nms"]["iou_threshold"],
#     )
#
#     total_boxes_before_nms += boxes_before_nms
#     total_boxes_after_nms += len(boxes)
#
#     print(f"{idx}/{len(val_samples)}")
#     #print("image:", image_path)
#     # print(
#     #     "number of detections after NMS:",
#     #     len(boxes),
#     #     "from:",
#     #     boxes_before_nms,
#     #     "vs ground truth:",
#     #     len(true_bboxes),
#     # )
#     # print("number of detections:", len(boxes), "vs ground truth:", len(true_bboxes))
#
#     matches = match_true_boxes_with_predictions(
#         true_bboxes=true_bboxes,
#         pred_boxes=boxes,
#         pred_confidence_scores=confidence_scores
#     )
#     for match in matches:
#
#         iou = match["iou"]
#
#         total_true_boxes += 1
#         sum_iou += iou
#
#
#         # print(
#         #     "true box idx:", match["true_box_idx"],
#         #     "best: "
#         #     "confidence_score:", match["best_confidence_score"],
#         #     "IoU:", match["iou"],
#         # )
#
#     #drew_bbox_and_save(image_path, boxes, true_bboxes, config)
#
#
# avg_boxes_before_nms = (
#     total_boxes_before_nms / len(val_samples)
#     if len(val_samples) > 0
#     else 0.0
# )
#
# avg_boxes_after_nms = (
#     total_boxes_after_nms / len(val_samples)
#     if len(val_samples) > 0
#     else 0.0
# )
#
# val_metrics = {
#     "val_mean_iou": mean_iou(sum_iou, total_true_boxes),
#     "val_avg_boxes_before_nms": avg_boxes_before_nms,
#     "val_avg_boxes_after_nms": avg_boxes_after_nms,
# }
#
# wandb_log(pipeline_run, val_metrics)
# finish_wandb(pipeline_run)
#
# print(f'mean_iou = {val_metrics["val_mean_iou"]}')
test_image_visualize(full_dataset, detector, non_max_suppression, config)