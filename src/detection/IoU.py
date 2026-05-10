def box_area(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = x1 - x2
    return width * height

def box_iou(bbox_pred, bbox_true):
    x1_pred, y1_pred, x2_pred, y2_pred = bbox_pred
    x1_true, y1_true, x2_true, y2_true = bbox_true

    inter_x1 = max(x1_pred, x1_true)
    inter_y1 = max(y1_pred, y1_true)
    inter_x2 = min(x2_pred, x2_true)
    inter_y2 = min(y2_pred, y2_true)

    intersection = box_area((inter_x1, inter_y1, inter_x2, inter_y2))

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