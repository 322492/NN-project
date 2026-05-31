import torch
import lightning as L

from src.models.baseline.sliding_window_detection import SlidingWindow
from src.detection.nms import non_max_suppression
from src.detection.detection_metrics import (
    evaluate_detections,
    calculate_map,
    calculate_precision_recall_f1,
)
from src.detection.IoU import match_true_boxes_with_predictions, mean_iou


class LitSlidingWindowCNNDetector(L.LightningModule):
    def __init__(self, cnn, config):
        super().__init__()

        self.cnn = cnn
        self.config = config
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.save_hyperparameters(ignore=["cnn"])

        self.val_detector_stats = None
        self.test_detector_stats = None

    def forward(self, crops):
        return self.cnn(crops)

    def _shared_crop_step(self, batch, stage: str):
        crops, labels = batch

        logits = self(crops).squeeze(1)
        loss = self.loss_fn(logits, labels.float())

        threshold = self.config["cnn_training"]["threshold"]
        scores = torch.sigmoid(logits)
        preds = scores >= threshold
        acc = (preds == labels.bool()).float().mean()

        self.log(
            f"cnn_{stage}_loss",
            loss,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        self.log(
            f"cnn_{stage}_acc",
            acc,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_crop_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        #to na cropach -> trening CNN
        if dataloader_idx == 0:
            return self._shared_crop_step(batch, "val")

        #to n pełnych zdj -> sliding window + CNN + NMS
        if dataloader_idx == 1:
            sample = batch[0]
            self._detector_step(sample, stage="val")
            return None

        raise ValueError(f"Unknown validation dataloader_idx: {dataloader_idx}")

    def test_step(self, batch, batch_idx):
        #pełne obrazy

        sample = batch[0]
        self._detector_step(sample, stage="test")
        return None

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.cnn.parameters()),
            lr=self.config["cnn_training"]["learning_rate"],
        )

    def build_detector(self, device):
        return SlidingWindow(
            cnn=self.cnn,
            window_sizes=tuple(self.config["sliding_window"]["window_sizes"]),
            overlap_ratio=self.config["sliding_window"]["overlap_ratio"],
            threshold=self.config["sliding_window"]["threshold"],
            crop_size=self.config["cnn_dataset"]["crop_size"],
            device=device,
        )

    def _new_detector_stats(self):
        return {
            "images": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_true_boxes": 0,
            "sum_iou": 0.0,
            "total_boxes_before_nms": 0,
            "total_boxes_after_nms": 0,
            "all_pred_boxes": [],
            "all_pred_scores": [],
            "all_true_boxes": [],
        }

    def on_validation_epoch_start(self):
        self.val_detector_stats = self._new_detector_stats()

    def on_test_epoch_start(self):
        self.test_detector_stats = self._new_detector_stats()

    def _detector_step(self, sample: dict, stage: str):
        if stage == "val":
            stats = self.val_detector_stats
        elif stage == "test":
            stats = self.test_detector_stats
        else:
            raise ValueError(f"Unknown stage: {stage}")

        image_path = sample["image_path"]
        true_bboxes = sample["bboxes"]

        detector = self.build_detector(device=self.device)

        with torch.no_grad():
            boxes, scores = detector.predict_window(image_path)

        boxes_before_nms = len(boxes)

        boxes, scores = non_max_suppression(
            boxes,
            scores,
            iou_threshold=self.config["nms"]["iou_threshold"],
        )

        metrics = evaluate_detections(
            pred_boxes=boxes,
            pred_scores=scores,
            true_boxes=true_bboxes,
            iou_threshold=self.config["metrics"]["iou_threshold"],
        )

        stats["images"] += 1
        stats["total_tp"] += metrics["tp"]
        stats["total_fp"] += metrics["fp"]
        stats["total_fn"] += metrics["fn"]

        stats["total_boxes_before_nms"] += boxes_before_nms
        stats["total_boxes_after_nms"] += len(boxes)

        stats["all_pred_boxes"].append(boxes)
        stats["all_pred_scores"].append(scores)
        stats["all_true_boxes"].append(true_bboxes)

        matches = match_true_boxes_with_predictions(
            true_bboxes=true_bboxes,
            pred_boxes=boxes,
            pred_confidence_scores=scores,
        )

        for match in matches:
            stats["total_true_boxes"] += 1
            stats["sum_iou"] += match["iou"]

    def _log_detector_epoch_metrics(self, stage: str):
        if stage == "val":
            stats = self.val_detector_stats
            prefix = "detector_val"
        elif stage == "test":
            stats = self.test_detector_stats
            prefix = "detector_test"
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if stats is None or stats["images"] == 0:
            return

        precision, recall, f1 = calculate_precision_recall_f1(
            stats["total_tp"],
            stats["total_fp"],
            stats["total_fn"],
        )

        map_score = calculate_map(
            stats["all_pred_boxes"],
            stats["all_pred_scores"],
            stats["all_true_boxes"],
            iou_threshold=self.config["metrics"]["iou_threshold"],
        )

        mean_iou_value = mean_iou(
            stats["sum_iou"],
            stats["total_true_boxes"],
        )

        avg_boxes_before_nms = (
            stats["total_boxes_before_nms"] / stats["images"]
            if stats["images"] > 0
            else 0.0
        )

        avg_boxes_after_nms = (
            stats["total_boxes_after_nms"] / stats["images"]
            if stats["images"] > 0
            else 0.0
        )

        self.log(f"{prefix}_precision", precision, prog_bar=False)
        self.log(f"{prefix}_recall", recall, prog_bar=False)
        self.log(f"{prefix}_f1", f1, prog_bar=True)
        self.log(f"{prefix}_map", map_score, prog_bar=True)
        self.log(f"{prefix}_mean_iou", mean_iou_value, prog_bar=True)
        self.log(f"{prefix}_avg_boxes_before_nms", avg_boxes_before_nms, prog_bar=False)
        self.log(f"{prefix}_avg_boxes_after_nms", avg_boxes_after_nms, prog_bar=False)
        self.log(f"{prefix}_tp", stats["total_tp"], prog_bar=False)
        self.log(f"{prefix}_fp", stats["total_fp"], prog_bar=False)
        self.log(f"{prefix}_fn", stats["total_fn"], prog_bar=False)

        print()
        print(f"{stage.upper()} DETECTOR SUMMARY")
        print("images:", stats["images"])
        print("TP:", stats["total_tp"])
        print("FP:", stats["total_fp"])
        print("FN:", stats["total_fn"])
        print("precision:", f"{precision:.4f}")
        print("recall:", f"{recall:.4f}")
        print("f1:", f"{f1:.4f}")
        print("mAP:", f"{map_score:.4f}")
        print("mean_iou:", f"{mean_iou_value:.4f}")
        print("avg boxes before NMS:", f"{avg_boxes_before_nms:.4f}")
        print("avg boxes after NMS:", f"{avg_boxes_after_nms:.4f}")
        print()

    def on_validation_epoch_end(self):
        self._log_detector_epoch_metrics(stage="val")

    def on_test_epoch_end(self):
        self._log_detector_epoch_metrics(stage="test")