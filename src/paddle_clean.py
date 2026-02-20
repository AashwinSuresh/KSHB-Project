import cv2
import numpy as np
from paddleocr import PaddleOCR
img_path = r"C:\@KHSB\sample\test_images\test02.png"
# -----------------------------
# IMAGE PREPROCESSING FUNCTION
# -----------------------------
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)

        # Adaptive threshold (better for handwriting)
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


# -----------------------------
# OCR INITIALIZATION
# -----------------------------
try:
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        rec=True,
        det=True,
        use_gpu=False  # Change to True if GPU available
    )
except Exception as e:
    print(f"OCR Initialization Error: {e}")
    exit()


# -----------------------------
# RUN OCR
# -----------------------------
img_path = 'test04.png'

processed_image = preprocess_image(img_path)

if processed_image is None:
    print("Image preprocessing failed.")
    exit()

try:
    result = ocr.ocr(processed_image, cls=True)
except Exception as e:
    print(f"OCR Error: {e}")
    exit()


# -----------------------------
# OUTPUT RESULTS
# -----------------------------
print("\n" + "="*40)
print("IMPROVED OCR RESULTS")
print("="*40)

if result and result[0]:
    for line in result[0]:
        text = line[1][0]
        confidence = line[1][1]

        # Filter low-confidence results
        if confidence > 0.50:   # adjust if needed
            print(f"{text}  (Confidence: {confidence:.2f})")
else:
    print("No text detected.")