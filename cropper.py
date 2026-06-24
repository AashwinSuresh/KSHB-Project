import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "model/dla-model.pt"

# Load model and force CPU usage
DETECTION_MODEL = YOLO(MODEL_PATH)
DETECTION_MODEL.to("cpu")
print(next(DETECTION_MODEL.model.parameters()).device)
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

HEADER_ANCHOR_CLASSES = [
    "List-item",
    "From",
    "Title",
    "Page-header"
]


def process_image(image_path, buffer_px=60):
    """
    Processes the image and crops the header metadata with relaxed padding.

    buffer_px: Increased default from 15 to 60 to prevent tight/partial cuts.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not load image")

    original_image = image.copy()
    boxed_image = image.copy()

    h, w = image.shape[:2]

    # Force prediction on CPU
    results = DETECTION_MODEL.predict(
        source=image,
        conf=0.2,
        iou=0.8,
        device="cpu",
        verbose=False
    )

    boxes = results[0].boxes

    max_y_header = 0
    header_found = False
    body_boxes = []

    class_names = list(ENTITIES_COLORS.keys())

    for box in boxes:
        cls_idx = int(box.cls)
        cls = class_names[cls_idx]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(
            boxed_image,
            (x1, y1),
            (x2, y2),
            ENTITIES_COLORS[cls],
            2
        )

        # Draw label
        cv2.putText(
            boxed_image,
            cls,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            ENTITIES_COLORS[cls],
            2
        )

        # Track header elements
        if cls in HEADER_ANCHOR_CLASSES:
            max_y_header = max(max_y_header, y2)
            header_found = True

        # Track body text regions
        if cls in ["Text", "Section-header"]:
            body_boxes.append((x1, y1, x2, y2))

    # Determine crop line
    if header_found:
        cutoff_y = min(max_y_header + buffer_px, h)

    elif body_boxes:
        body_boxes.sort(
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )

        largest_body_ymin = body_boxes[0][1]
        cutoff_y = max(0, largest_body_ymin - 20)

    else:
        cutoff_y = int(h * 0.50)

    # Crop image
    cropped = original_image[0:cutoff_y, 0:w]

    # Save crop
    os.makedirs("temp", exist_ok=True)
    cv2.imwrite("temp/cropped_region.jpg", cropped)

    return original_image, boxed_image, cropped