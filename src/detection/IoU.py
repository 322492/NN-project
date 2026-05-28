def box_area(bbox):
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def box_iou(bbox_pred, bbox_true):
    x1_pred, y1_pred, x2_pred, y2_pred = bbox_pred
    x1_true, y1_true, x2_true, y2_true = bbox_true

    inter_x1 = max(x1_pred, x1_true)
    inter_y1 = max(y1_pred, y1_true)
    inter_x2 = min(x2_pred, x2_true)
    inter_y2 = min(y2_pred, y2_true)

    intersection = box_area([inter_x1, inter_y1, inter_x2, inter_y2])

    area_pred = box_area(bbox_pred)
    area_true = box_area(bbox_true)

    union = area_pred + area_true - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def max_iou_with_true_boxes(candidate_bbox, true_bboxes):
    if len(true_bboxes) == 0:
        return 0.0

    max_iou = 0.0

    for true_bbox in true_bboxes:
        iou = box_iou(candidate_bbox, true_bbox)
        if iou > max_iou:
            max_iou = iou

    return max_iou

def match_true_boxes_with_predictions(true_bboxes, pred_boxes, pred_confidence_scores=None):
    results = []

    for true_idx, true_box in enumerate(true_bboxes):
        best_iou = 0.0
        best_pred_box = None
        best_confidence_score = None

        for pred_idx, pred_box in enumerate(pred_boxes):
            iou = box_iou(pred_box, true_box)

            if iou > best_iou:
                best_iou = iou
                best_pred_box = pred_box

                if pred_confidence_scores is not None:
                    best_confidence_score = pred_confidence_scores[pred_idx]

        results.append({
            "true_box_idx": true_idx,
            "true_box": true_box,
            "best_pred_box": best_pred_box,
            "best_confidence_score": best_confidence_score,
            "iou": best_iou,
        })

    return results

def mean_iou(sum_iou, total_true_bboxes):
    if total_true_bboxes == 0:
        return 0.0

    return sum_iou / total_true_bboxes
