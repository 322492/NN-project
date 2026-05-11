from src.detection.IoU import box_iou


def non_max_suppression(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return [], []

    sorted_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    kept_boxes = []
    kept_scores = []

    while len(sorted_indices) > 0:
        best_index = sorted_indices[0]
        best_box = boxes[best_index]
        best_score = scores[best_index]

        kept_boxes.append(best_box)
        kept_scores.append(best_score)

        remaining_indices = []

        for index in sorted_indices[1:]:
            iou = box_iou(best_box, boxes[index])

            if iou <= iou_threshold:
                remaining_indices.append(index)

        sorted_indices = remaining_indices

    return kept_boxes, kept_scores
