import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "model/dla-model.pt"

DETECTION_MODEL = YOLO(MODEL_PATH)

ENTITIES_COLORS = {
    "Caption": (191, 100, 21),
    "Footnote": (2, 62, 115),
    "Formula": (140, 80, 58),
    "List-item": (168, 181, 69),
    "Page-footer": (2, 69, 84),
    "Page-header": (83, 115, 106),
    "Picture": (255, 72, 88),
    "Section-header": (0, 204, 192),
    "Table": (116, 127, 127),
    "Text": (0, 153, 221),
    "Title": (196, 51, 2)
}

VALID_CLASSES = [
    "Text",
    "Section-header"
]


def process_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not load image")

    original_image = image.copy()
    boxed_image = image.copy()

    h, w = image.shape[:2]

    results = DETECTION_MODEL.predict(
        source=image,
        conf=0.2,
        iou=0.8
    )

    boxes = results[0].boxes

    selected_boxes = []

    for box in boxes:

        cls = list(ENTITIES_COLORS)[int(box.cls)]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(
            boxed_image,
            (x1, y1),
            (x2, y2),
            ENTITIES_COLORS[cls],
            2
        )

        cv2.putText(
            boxed_image,
            cls,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            ENTITIES_COLORS[cls],
            2
        )

        # Ignore lower 30%
        if y1 > h * 0.7:
            continue

        if cls in VALID_CLASSES:
            selected_boxes.append(
                (x1, y1, x2, y2)
            )

    if len(selected_boxes) == 0:
        return original_image, boxed_image, original_image

    min_x = min(box[0] for box in selected_boxes)
    min_y = min(box[1] for box in selected_boxes)
    max_x = max(box[2] for box in selected_boxes)
    max_y = max(box[3] for box in selected_boxes)

    cropped = image[min_y:max_y, min_x:max_x]

    os.makedirs("temp", exist_ok=True)

    cv2.imwrite(
        "temp/cropped_region.jpg",
        cropped
    )

    return original_image, boxed_image, cropped