from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False)

img_path = r"C:\@KHSB\sample\test_images\test02.png"

# Run OCR
result = ocr.ocr(img_path, cls=True)

print("\n" + "="*30)
print("EXTRACTED TEXT RESULTS")
print("="*30)

if result:
   for i in range(len(result[0])):
    print(result[0][i][1][0])
else:
    print("No text detected.")


