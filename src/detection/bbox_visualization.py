from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path

def draw_boxes_on_image(image_path, pred_boxes, true_boxes, output_path=None, pred_color="orange", true_color="magenta", width=4):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in true_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline=true_color, width=width+2)

    for box in pred_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline=pred_color, width=width)

    if output_path is not None:
        image.save(output_path)

    return image

def drew_bbox_and_save(image_path, boxes, true_boxes, config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    result_image_path = config["outputs"]["result_image_path"].format(timestamp)

    Path(result_image_path).parent.mkdir(parents=True, exist_ok=True)

    draw_boxes_on_image(
        image_path,
        boxes,
        true_boxes,
        result_image_path,
    )

def test_image_visualize(full_dataset, detector, non_max_suppression, config):
    # konkretne zdj
    DEBUG_IMAGE_NAME = "960.jpg"
    debug_sample = None

    for sample in full_dataset.samples:
        if sample["file_name"] == DEBUG_IMAGE_NAME:
            debug_sample = sample
            break

    if debug_sample is None:
        if len(full_dataset.samples) == 0:
            print(f"Skipping visualization: no images in dataset.")
            return
        debug_sample = full_dataset.samples[0]
        print(
            f"Debug image {DEBUG_IMAGE_NAME} not in dataset, "
            f"using {debug_sample['file_name']} instead."
        )

    image_path = debug_sample["image_path"]
    true_bboxes = debug_sample["bboxes"]

    boxes, scores = detector.predict_window(image_path)
    boxes_before_nms = len(boxes)
    boxes, scores = non_max_suppression(
        boxes,
        scores,
        iou_threshold=config["nms"]["iou_threshold"],
    )

    drew_bbox_and_save(image_path, boxes, true_bboxes, config)