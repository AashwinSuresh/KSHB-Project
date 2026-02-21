import cv2
import numpy as np
import re
from paddleocr import PaddleOCR


# ---------------------------------------------------
# IMAGE PATH
# ---------------------------------------------------
# img_path = r"C:\@KHSB\sample\test_images\test_secretary.png"
img_path = r"C:\@KHSB\sample\test_images\test_chairperson.png"
# img_path = r"C:\@KHSB\sample\test_images\test_chief_engineer.png"


# ---------------------------------------------------
# IMAGE PREPROCESSING FUNCTION
# ---------------------------------------------------
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Image not found or invalid path.")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove noise
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        return thresh

    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None


# ---------------------------------------------------
# EXTRACT ONLY TO SECTION (METHOD B + STRUCTURE FILTER)
# ---------------------------------------------------
def extract_to_section(result, height_ratio=0.45):
    if not result or not result[0]:
        return ""

    # Collect Y values
    all_y = [point[1] for line in result[0] for point in line[0]]
    max_height = max(all_y)
    height_limit = max_height * height_ratio

    # Sort lines top to bottom
    sorted_lines = sorted(result[0], key=lambda x: x[0][0][1])

    top_region_lines = []

    for line in sorted_lines:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]

        if confidence < 0.5:
            continue

        y_avg = sum([p[1] for p in box]) / 4

        if y_avg <= height_limit:
            top_region_lines.append(text.strip())

    # Extract after "To"
    to_section = ""
    capture = False

    for line in top_region_lines:
        lower_line = line.lower().strip()

        # Detect "To", "To:", "TO:", etc.
        if lower_line.startswith("to"):
            capture = True
            continue

        if capture:
            # Stop when subject appears
            if lower_line.startswith("subject"):
                break
            to_section += lower_line + " "

    return to_section

# ---------------------------------------------------
# CLASSIFICATION FUNCTION
# ---------------------------------------------------
def classify_letter(text):
    text = text.lower()

    if re.search(r"chief\s*engineer", text):
        return "chief_engineer"

    elif re.search(r"secretary", text):
        return "secretary"

    elif re.search(r"chair\s*person", text):
        return "chair_person"

    else:
        return "errroorrrrr........"


# ---------------------------------------------------
# OCR INITIALIZATION
# ---------------------------------------------------
try:
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        show_log=False
    )
except Exception as e:
    print(f"OCR Initialization Error: {e}")
    exit()


# ---------------------------------------------------
# RUN OCR PIPELINE
# ---------------------------------------------------
processed_image = preprocess_image(img_path)

if processed_image is None:
    print("Image preprocessing failed.")
    exit()

try:
    result = ocr.ocr(processed_image, cls=True)
except Exception as e:
    print(f"OCR Error: {e}")
    exit()


# ---------------------------------------------------
# PRINT RAW OCR (Optional Debug)
# ---------------------------------------------------
print("\n" + "=" * 40)
print("RAW OCR TEXT")
print("=" * 40)

if result and result[0]:
    for line in result[0]:
        text = line[1][0]
        confidence = line[1][1]
        if confidence > 0.5:
            print(f"{text} (Confidence: {confidence:.2f})")
else:
    print("No text detected.")


# ---------------------------------------------------
# EXTRACT TO SECTION
# ---------------------------------------------------
to_section_text = extract_to_section(result, height_ratio=0.45)

print("\n" + "=" * 40)
print("EXTRACTED TO SECTION")
print("=" * 40)
print(to_section_text)


# ---------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------
category = classify_letter(to_section_text)

print("\n" + "=" * 40)
print("CLASSIFICATION RESULT")
print("=" * 40)
print("Category:", category)