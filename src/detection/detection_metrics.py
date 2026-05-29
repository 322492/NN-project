from src.detection.IoU import box_iou

# match one predicted box to the best unmatched ground-truth box if IoU meets the threshold
def _match_prediction(pred_box, true_boxes, matched_true_indices, iou_threshold=0.5):
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
        return True, best_true_index

    return False, None

# count TP, FP, and FN for one image by matching score-sorted predictions to ground truth
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
        is_true_positive, best_true_index = _match_prediction(
            pred_box,
            true_boxes,
            matched_true_indices,
            iou_threshold=iou_threshold,
        )

        if is_true_positive:
            true_positive += 1
            matched_true_indices.add(best_true_index)
        else:
            false_positive += 1

    false_negative = len(true_boxes) - len(matched_true_indices)

    return true_positive, false_positive, false_negative

# compute precision, recall, and F1 from TP, FP, and FN counts
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

# return TP, FP, FN, precision, recall, and F1 for a single image’s detections
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

# compute mAP over a dataset from lists of per-image predictions and ground truth
def calculate_map(pred_boxes_list, pred_scores_list, true_boxes_list, iou_threshold=0.5):
    detections = []

    for image_index, (pred_boxes, pred_scores) in enumerate(
        zip(pred_boxes_list, pred_scores_list)
    ):
        for pred_index, pred_box in enumerate(pred_boxes):
            detections.append(
                (pred_scores[pred_index], image_index, pred_box)
            )

    detections.sort(key=lambda detection: detection[0], reverse=True)

    num_ground_truth = sum(len(true_boxes) for true_boxes in true_boxes_list)
    if num_ground_truth == 0:
        return 0.0

    matched_ground_truth = [set() for _ in true_boxes_list]
    true_positive_count = 0
    false_positive_count = 0
    recalls = []
    precisions = []

    for _, image_index, pred_box in detections:
        true_boxes = true_boxes_list[image_index]
        is_true_positive, best_true_index = _match_prediction(
            pred_box,
            true_boxes,
            matched_ground_truth[image_index],
            iou_threshold=iou_threshold,
        )

        if is_true_positive:
            true_positive_count += 1
            matched_ground_truth[image_index].add(best_true_index)
        else:
            false_positive_count += 1

        false_negative_count = num_ground_truth - true_positive_count
        precision, recall, _ = calculate_precision_recall_f1(
            true_positive_count,
            false_positive_count,
            false_negative_count,
        )
        recalls.append(recall)
        precisions.append(precision)

    if not recalls:
        return 0.0

    average_precision = 0.0
    previous_recall = 0.0

    for step_index, recall in enumerate(recalls):
        if recall <= previous_recall:
            continue

        interpolated_precision = max(
            precisions[other_index]
            for other_index in range(len(precisions))
            if recalls[other_index] >= recall
        )
        average_precision += (recall - previous_recall) * interpolated_precision
        previous_recall = recall

    return average_precision
