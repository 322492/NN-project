from src.detection.IoU import box_iou


def count_detection_results(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
    matched_true_indices = set()
    true_positive = 0
    false_positive = 0

    sorted_indices = sorted(
        range(len(pred_scores)),
        key=lambda index: pred_scores[index],
        reverse=True,
    )

    for pred_index in sorted_indices:
        pred_box = pred_boxes[pred_index]

        best_iou = 0.0
        best_true_index = None

        for true_index, true_box in enumerate(true_boxes):
            if true_index in matched_true_indices:
                continue

            iou = box_iou(pred_box, true_box)

            if iou > best_iou:
                best_iou = iou
                best_true_index = true_index

        if best_iou >= iou_threshold:
            true_positive += 1
            matched_true_indices.add(best_true_index)
        else:
            false_positive += 1

    false_negative = len(true_boxes) - len(matched_true_indices)

    return true_positive, false_positive, false_negative


def calculate_precision_recall_f1(true_positive, false_positive, false_negative):
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)

    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (true_positive + false_negative)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_detections(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
    true_positive, false_positive, false_negative = count_detection_results(
        pred_boxes,
        pred_scores,
        true_boxes,
        iou_threshold=iou_threshold,
    )

    precision, recall, f1 = calculate_precision_recall_f1(
        true_positive,
        false_positive,
        false_negative,
    )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
