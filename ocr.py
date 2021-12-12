import cv2
import pytesseract
import numpy as np
from deep_translator import GoogleTranslator


pytesseract.pytesseract.tesseract_cmd = 'D:\\Program Files\\Tesseract-OCR\\tesseract'


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Preprocessing the image
img = cv2.imread("a-Telugu-text-image.png")
img = cv2.resize(img, None, fx=1, fy=1)  # resizing image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale color

# reducing Noise
# adaptive_threshold = cv2.adaptiveThreshold(
#     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 95, 11)


threshold = thresholding(gray)

# scaning image and converting to text
config = "-l tel --psm 6"  # page segmentation mode - psm
text = pytesseract.image_to_string(threshold, config=config)

# translating text
translated = GoogleTranslator(
    source='auto', target='en').translate(text)

print(text, translated)

# Image display
cv2.imshow("Threshold", threshold)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
