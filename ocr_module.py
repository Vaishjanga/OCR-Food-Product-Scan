import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
import pytesseract
import os
import spacy
from rapidfuzz import fuzz, process
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load spaCy model globally
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# Enhanced fuzzy keyword lists with common OCR variations
FUZZY_KEYWORDS = {
    'ingredients': [
        'ingredients', 'ingredlents', 'ingredlents:', 'lnqredients', 'ingred.', 
        'ingredlents', 'composition', 'contains:', 'made with:', 'formula:',
        'ingredienls', 'ingredienis', 'ingrecients', 'ingredents'
    ],
    'nutrition': [
        'nutrition facts', 'nutritional information', 'nutritlon', 'nutrltlon', 
        'nutritional value', 'nutritional', 'nutrition', 'nutritional facts',
        'nutrition information', 'per serving:', 'per 100g:', 'per 100ml:',
        'nutrition panel', 'nutrltion', 'nutritlonal', 'nutrltlonal'
    ],
    'brand': [
        'brand', 'manufacturer', 'company', 'made by', 'produced by',
        'distributed by', 'marketed by', 'trademark', 'brand name'
    ],
    'categories': [
        'category', 'categories', 'type', 'product type', 'food type',
        'product category', 'classification'
    ]
}

# Common nutrition keywords for better extraction
NUTRITION_KEYWORDS = [
    'calories', 'energy', 'protein', 'fat', 'saturated fat', 'trans fat',
    'cholesterol', 'sodium', 'carbohydrate', 'sugar', 'fiber', 'calcium',
    'iron', 'vitamin', 'kcal', 'kj', 'carbs', 'fibre'
]

def enhanced_fuzzy_find_keyword(line, keyword_list, threshold=70):
    """Enhanced fuzzy matching with better scoring"""
    line_lower = line.lower().strip()
    best_score = 0
    best_match = None
    
    for kw in keyword_list:
        # Try exact match first
        if kw.lower() in line_lower:
            return True
        
        # Fuzzy matching with multiple algorithms
        scores = [
            fuzz.partial_ratio(kw.lower(), line_lower),
            fuzz.token_set_ratio(kw.lower(), line_lower),
            fuzz.ratio(kw.lower(), line_lower)
        ]
        max_score = max(scores)
        
        if max_score >= threshold and max_score > best_score:
            best_score = max_score
            best_match = kw
    
    return best_score >= threshold

def advanced_preprocess_image(image_path, target_method='combined'):
    """
    Multiple preprocessing strategies for different OCR engines
    """
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]
    
    preprocessed_images = []
    
    # Strategy 1: Standard preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize if needed
    height, width = gray.shape
    if width < 1000:
        scale = 1000 / width
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Multiple thresholding approaches
    thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    
    # Strategy 2: Different thresholding
    thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Strategy 3: Morphological operations
    kernel = np.ones((1,1), np.uint8)
    morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    
    # Save different versions
    temp_paths = []
    for i, processed in enumerate([thresh1, thresh2, morph1]):
        temp_path = f"preprocessed_temp_{i}.jpg"
        cv2.imwrite(temp_path, processed)
        temp_paths.append(temp_path)
        preprocessed_images.append(temp_path)
    
    return preprocessed_images

def extract_text_multiple_methods(image_path):
    """
    Extract text using multiple preprocessing and OCR combinations
    """
    preprocessed_images = advanced_preprocess_image(image_path)
    all_results = []
    
    # Tesseract configurations
    tesseract_configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 4',
        r'--oem 3 --psm 11',
        r'--oem 1 --psm 6'
    ]
    
    # Try each preprocessing with each config
    for prep_img in preprocessed_images:
        for config in tesseract_configs:
            try:
                text = pytesseract.image_to_string(Image.open(prep_img), config=config)
                if text.strip():
                    all_results.append(clean_text(text))
            except Exception as e:
                logger.warning(f"Tesseract failed with config {config}: {e}")
        
        # EasyOCR
        try:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            result = reader.readtext(prep_img, detail=1, paragraph=True)
            confident_results = [r[1] for r in result if r[2] > 0.3]
            if confident_results:
                all_results.append(clean_text(" ".join(confident_results)))
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
    
    # Clean up temp files
    for temp_path in preprocessed_images:
        if temp_path != image_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
    # Return the best result (longest meaningful text)
    if all_results:
        # Score results by length and word count
        scored_results = []
        for text in all_results:
            words = len(text.split())
            chars = len(text)
            score = words * 2 + chars  # Prefer results with more words
            scored_results.append((score, text))
        
        best_result = max(scored_results, key=lambda x: x[0])[1]
        return best_result
    
    return ""

def clean_text(text):
    """Enhanced text cleaning"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors
    ocr_fixes = {
        r'\b0(?=\w)': 'O',  # 0 -> O at word start
        r'(?<=\w)0\b': 'O',  # 0 -> O at word end
        r'\bl(?=\w)': 'I',   # l -> I in some cases
        r'rn': 'm',          # rn -> m
        r'cl': 'd',          # cl -> d
        r'vv': 'w',          # vv -> w
    }
    
    for pattern, replacement in ocr_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove obviously wrong characters
    text = re.sub(r'[^\w\s.,;:()%-/&\[\]{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def advanced_parse_ocr_text(text):
    """
    Advanced parsing with better field extraction
    """
    if not text:
        return {
            'product_name': 'N/A',
            'ingredients_text': 'N/A',
            'nutriments': {},
            'brands': 'N/A',
            'categories': 'N/A',
        }
    
    text = clean_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 1]
    
    # Initialize results
    result = {
        'product_name': 'N/A',
        'ingredients_text': 'N/A',
        'nutriments': {},
        'brands': 'N/A',
        'categories': 'N/A',
    }
    
    # Extract product name - look for title-like text at the beginning
    for i, line in enumerate(lines[:5]):
        # Skip lines that look like headers or labels
        if any(enhanced_fuzzy_find_keyword(line, FUZZY_KEYWORDS[key], 70) 
               for key in FUZZY_KEYWORDS):
            continue
        
        # Look for lines that seem like product names
        if (len(line) > 5 and 
            len(line.split()) <= 8 and  # Not too long
            not re.match(r'^\d+.*', line) and  # Doesn't start with number
            not line.isupper()):  # Not all caps (likely labels)
            result['product_name'] = line
            break
    
    # Extract ingredients with context awareness
    ingredients_found = False
    for i, line in enumerate(lines):
        if enhanced_fuzzy_find_keyword(line, FUZZY_KEYWORDS['ingredients'], 65):
            ingredients_parts = []
            
            # Check if ingredients are on the same line
            colon_split = line.split(':', 1)
            if len(colon_split) > 1 and len(colon_split[1].strip()) > 10:
                ingredients_parts.append(colon_split[1].strip())
            
            # Look for ingredients in following lines
            for j in range(i+1, min(i+15, len(lines))):
                next_line = lines[j]
                
                # Stop if we hit another section
                if any(enhanced_fuzzy_find_keyword(next_line, FUZZY_KEYWORDS[key], 70) 
                       for key in ['nutrition', 'brand', 'categories']):
                    break
                
                # Skip very short lines or lines that look like labels
                if len(next_line) < 3 or next_line.isupper():
                    continue
                
                ingredients_parts.append(next_line)
                
                # Stop after reasonable amount of text
                if len(' '.join(ingredients_parts)) > 300:
                    break
            
            if ingredients_parts:
                result['ingredients_text'] = ' '.join(ingredients_parts)
                ingredients_found = True
                break
    
    # Extract nutrition information
    nutrition_data = {}
    for i, line in enumerate(lines):
        if enhanced_fuzzy_find_keyword(line, FUZZY_KEYWORDS['nutrition'], 65):
            # Look for nutrition values in surrounding lines
            for j in range(max(0, i-2), min(i+20, len(lines))):
                nut_line = lines[j]
                
                # More comprehensive nutrition pattern
                patterns = [
                    r'([A-Za-z\s]+?)\s*[:\-]\s*([0-9]+\.?[0-9]*)\s*(g|mg|kcal|kj|%)?',
                    r'([A-Za-z\s]+?)\s+([0-9]+\.?[0-9]*)\s*(g|mg|kcal|kj|%)',
                    r'([A-Za-z\s]*(?:' + '|'.join(NUTRITION_KEYWORDS) + r')[A-Za-z\s]*)\s+([0-9]+\.?[0-9]*)'
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, nut_line, re.IGNORECASE)
                    for match in matches:
                        key = match.group(1).strip().title()
                        value = match.group(2)
                        unit = match.group(3) if len(match.groups()) > 2 and match.group(3) else ''
                        
                        # Clean up the key
                        key = re.sub(r'[^\w\s]', '', key).strip()
                        if len(key) > 2:
                            nutrition_data[key] = f"{value}{unit}"
            break
    
    result['nutriments'] = nutrition_data
    
    # Extract brand with better patterns
    for i, line in enumerate(lines):
        if enhanced_fuzzy_find_keyword(line, FUZZY_KEYWORDS['brand'], 70):
            # Look for brand in same line or next line
            brand_candidates = [line]
            if i + 1 < len(lines):
                brand_candidates.append(lines[i + 1])
            
            for candidate in brand_candidates:
                brand_match = re.search(r'(?:brand|by|manufacturer)[:\s]+([A-Za-z0-9\s&\-\.]+)', 
                                      candidate, re.IGNORECASE)
                if brand_match:
                    result['brands'] = brand_match.group(1).strip()
                    break
            
            if result['brands'] != 'N/A':
                break
    
    # If no explicit brand found, use NLP to find organization entities
    if result['brands'] == 'N/A':
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and len(ent.text.strip()) > 2:
                    result['brands'] = ent.text.strip()
                    break
        except Exception:
            pass
    
    # Extract categories
    for line in lines:
        if enhanced_fuzzy_find_keyword(line, FUZZY_KEYWORDS['categories'], 70):
            cat_match = re.search(r'(?:category|type)[:\s]+([A-Za-z0-9\s,&\-]+)', 
                                line, re.IGNORECASE)
            if cat_match:
                result['categories'] = cat_match.group(1).strip()
                break
    
    return result

def extract_product_info(image_path):
    """
    Main function to extract all product information
    """
    logger.info(f"Processing image: {image_path}")
    
    # Extract text using multiple methods
    extracted_text = extract_text_multiple_methods(image_path)
    
    if not extracted_text:
        logger.warning("No text extracted from image")
        return {
            'product_name': 'N/A',
            'ingredients_text': 'N/A',
            'nutriments': {},
            'brands': 'N/A',
            'categories': 'N/A',
            'barcode': None,
            'extracted_text': ''
        }
    
    # Parse the extracted text
    parsed_info = advanced_parse_ocr_text(extracted_text)
    
    # Try to extract barcode
    try:
        barcode = extract_barcode(image_path)
        parsed_info['barcode'] = barcode
    except Exception as e:
        logger.warning(f"Barcode extraction failed: {e}")
        parsed_info['barcode'] = None
    
    # Add raw text for debugging
    parsed_info['extracted_text'] = extracted_text
    
    return parsed_info

def extract_barcode(image_path):
    """Enhanced barcode extraction"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Try multiple preprocessing approaches for barcode
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different image processing techniques
    processed_images = [
        gray,
        cv2.GaussianBlur(gray, (3, 3), 0),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ]
    
    for processed in processed_images:
        try:
            detector = cv2.barcode_BarcodeDetector()
            result = detector.detectAndDecode(processed)
            
            if len(result) >= 3 and result[0] and len(str(result[0])) >= 8:
                return str(result[0])
        except Exception:
            continue
    
    # Fallback: extract from OCR text
    try:
        text = pytesseract.image_to_string(gray, config='--psm 8 -c tessedit_char_whitelist=0123456789')
        matches = re.findall(r'\b\d{8,13}\b', text)
        if matches:
            return matches[0]
    except Exception:
        pass
    
    return None

# Example usage
if __name__ == "__main__":
    # Test the enhanced extractor
    image_path = "product_image.jpg"  # Replace with your image path
    
    result = extract_product_info(image_path)
    
    print("Extracted Product Information:")
    print(f"Product Name: {result['product_name']}")
    print(f"Brand: {result['brands']}")
    print(f"Categories: {result['categories']}")
    print(f"Ingredients: {result['ingredients_text']}")
    print(f"Nutrition: {result['nutriments']}")
    print(f"Barcode: {result['barcode']}")
    print(f"\nRaw extracted text:\n{result['extracted_text']}")