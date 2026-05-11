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