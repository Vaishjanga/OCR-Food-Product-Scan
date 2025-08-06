import os
print("TESSDATA_PREFIX:", os.environ.get('TESSDATA_PREFIX'))
print("PATH contains Tesseract:", 'Tesseract-OCR' in os.environ.get('PATH', ''))

import pytesseract
print("Tesseract version:", pytesseract.get_tesseract_version())