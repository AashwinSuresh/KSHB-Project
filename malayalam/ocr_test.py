from paddleocr import PaddleOCR
import os

IMAGE_PATH = r"C:\@KHSB\sample\malayalam\malayalam_sample02.png"

print("Checking image...")

if not os.path.exists(IMAGE_PATH):
    print("Image not found!")
    exit()

print("Image found!")

print("Loading OCR model...")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

print("OCR model loaded!")

print("Running OCR...")

results = ocr.ocr(IMAGE_PATH)

print("\nRESULTS:\n")

print("\nRESULTS:\n")

for res in results:
    print("Recognized Texts:")
    print(res["rec_texts"])

    print("\nConfidence Scores:")
    print(res["rec_scores"])

    print("\nBounding Boxes:")
    print(res["rec_boxes"])