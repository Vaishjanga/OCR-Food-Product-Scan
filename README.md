# OCR-Based Product Information Scanner

## Overview
This project uses Optical Character Recognition (OCR) to scan product labels from images and retrieves product details from the Open Food Facts database. It features a simple Streamlit web interface for uploading images and displaying results.

## Features
- Extracts text and barcode from product images using EasyOCR and pyzbar.
- Queries Open Food Facts API for product details (name, brand, ingredients, nutrition, categories).
- User-friendly web interface built with Streamlit.

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
4. **Upload a product image** (JPG/PNG) via the web interface.

## File Structure
- `ocr_module.py`: OCR and barcode extraction functions
- `openfoodfacts_api.py`: API query functions
- `app.py`: Streamlit web app
- `requirements.txt`: Python dependencies
- `sample_images/`: Place your sample product images here

## Approach
- Uses EasyOCR to extract text from images.
- Uses pyzbar to extract barcodes (if present).
- Tries to match barcode first; if not found, falls back to product name search.
- Displays extracted text, barcode, and product details in the web interface.

## Notes
- For best results, use clear, high-resolution images of product labels.
- If no barcode is found, the app will attempt to match using the extracted text.

## Demo
Add screenshots or a short video demo in this section.