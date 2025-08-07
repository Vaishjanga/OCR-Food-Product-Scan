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
from textblob import TextBlob

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
# 3. Fix fuzzy matching (around line 60-80)
def enhanced_fuzzy_find_keyword(line, keyword_list, threshold=80):  # CHANGED: Increase from 70 to 80
    """Enhanced fuzzy matching with better scoring"""
    line_lower = line.lower().strip()
    
    # Try exact match first
    for kw in keyword_list:
        if kw.lower() in line_lower:
            return True
    
    # ADDED: Check if line starts with keyword (common case)
    for kw in keyword_list:
        if line_lower.startswith(kw.lower()):
            return True
    
    best_score = 0
    for kw in keyword_list:
        # Fuzzy matching with multiple algorithms
        scores = [
            fuzz.partial_ratio(kw.lower(), line_lower),
            fuzz.token_set_ratio(kw.lower(), line_lower),
        ]
        max_score = max(scores)
        
        if max_score > best_score:
            best_score = max_score
    
    return best_score >= threshold

def advanced_preprocess_image(image_path, target_method='combined'):
    """
    Completely different preprocessing approach for very poor quality images
    """
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]
    
    preprocessed_images = []

    # Much more aggressive upscaling
    height, width = img.shape[:2]
    target_width = 3000  # Increased significantly
    if width < target_width:
        scale = target_width / float(width)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Multiple different preprocessing strategies
    
    # Strategy 1: High contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    high_contrast = clahe.apply(gray)
    
    # Strategy 2: Gaussian blur + threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_blur = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Strategy 3: Median filter + adaptive threshold
    median = cv2.medianBlur(gray, 5)
    adaptive = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    
    # Strategy 4: Opening morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    
    # Strategy 5: Erosion + Dilation
    eroded = cv2.erode(thresh_blur, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    versions = [gray, high_contrast, thresh_blur, adaptive, opening, dilated]
    for i, version in enumerate(versions):
        temp_path = f"preprocessed_temp_{i}.jpg"
        cv2.imwrite(temp_path, version)
        preprocessed_images.append(temp_path)

    return preprocessed_images

def extract_any_readable_text(image_path):
    """
    Try every possible method to extract ANY readable text
    """
    all_text_results = []
    
    try:
        # Method 1: Try EasyOCR (often better with poor images)
        reader = easyocr.Reader(['en'])
        easyocr_result = reader.readtext(image_path)
        if easyocr_result:
            easyocr_text = ' '.join([result[1] for result in easyocr_result if result[2] > 0.3])  # Confidence > 0.3
            if easyocr_text.strip():
                all_text_results.append(('EasyOCR', easyocr_text))
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
    
    # Method 2: Multiple Tesseract attempts
    preprocessed_images = advanced_preprocess_image(image_path)
    
    tesseract_configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 8',  # Single word
        r'--oem 3 --psm 7',  # Single text line
        r'--oem 3 --psm 13', # Raw line
        r'--oem 1 --psm 6',  # Legacy engine
        r'--dpi 300 --psm 6',
    ]

    for prep_img in preprocessed_images:
        for config in tesseract_configs:
            try:
                text = pytesseract.image_to_string(Image.open(prep_img), config=config)
                if text.strip():
                    all_text_results.append((f'Tesseract-{config[:10]}', text.strip()))
            except Exception:
                continue

    # Clean up temp files
    for temp_path in preprocessed_images:
        if temp_path != image_path:
            safe_remove_file(temp_path)

    # Method 3: Try to extract individual characters/words with high confidence
    try:
        data = pytesseract.image_to_data(image_path, output_type=pytesseract.Output.DICT)
        high_conf_words = []
        for i, conf in enumerate(data['conf']):
            if int(conf) > 60:  # High confidence words only
                word = data['text'][i].strip()
                if len(word) > 2:
                    high_conf_words.append(word)
        
        if high_conf_words:
            high_conf_text = ' '.join(high_conf_words)
            all_text_results.append(('High-Conf', high_conf_text))
    except Exception:
        pass

    return all_text_results


def find_best_readable_content(all_results):
    """
    Find the most readable content from all OCR attempts
    """
    if not all_results:
        return ""
    
    scored_results = []
    
    for method, text in all_results:
        if not text or len(text.strip()) < 10:
            continue
            
        # Score based on readability
        words = text.split()
        if not words:
            continue
            
        readable_words = 0
        total_chars = 0
        
        for word in words:
            if len(word) >= 3 and word.replace('-', '').isalnum():
                # Check if word has reasonable vowel/consonant ratio
                vowels = sum(1 for c in word.lower() if c in 'aeiou')
                if len(word) <= 4 or vowels > 0:  # Short words OK, long words need vowels
                    readable_words += 1
                    total_chars += len(word)
        
        if len(words) > 0:
            readability_score = (readable_words / len(words)) * 100
            content_score = readable_words * 2
            char_score = total_chars / 10
            
            total_score = readability_score + content_score + char_score
            scored_results.append((total_score, method, text))
    
    if scored_results:
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][2]  # Return best text
    
    return ""

def safe_remove_file(file_path):
    """Safely remove a file with multiple attempts"""
    if not os.path.exists(file_path):
        return True
    
    import time
    import gc
    
    # Force garbage collection to close any file handles
    gc.collect()
    
    for attempt in range(3):
        try:
            os.remove(file_path)
            return True
        except PermissionError:
            if attempt < 2:
                time.sleep(0.2)
            else:
                # If still can't delete, try to rename for later cleanup
                try:
                    import random
                    cleanup_name = f"cleanup_{random.randint(1000, 9999)}.tmp"
                    os.rename(file_path, cleanup_name)
                    return True
                except:
                    return False
        except Exception:
            return False
    
    return False

def extract_text_multiple_methods(image_path):
    """
    Try every possible OCR method and return the best result
    """
    logger.info(f"Attempting multiple OCR methods on: {image_path}")
    
    all_results = extract_any_readable_text(image_path)
    best_text = find_best_readable_content(all_results)
    
    if not best_text:
        # Last resort: return the longest result even if poor quality
        if all_results:
            longest = max(all_results, key=lambda x: len(x[1]))
            logger.warning("Using poor quality OCR result as last resort")
            return longest[1]
    
    return best_text


def smart_parse_poor_ocr(text):
    """
    Specialized parsing for very poor OCR results
    """
    result = {
        'product_name': 'N/A',
        'ingredients_text': 'N/A',
        'nutriments': {},
        'brands': 'N/A',
        'categories': 'N/A',
    }
    
    if not text:
        return result
    
    # Convert to lowercase for easier pattern matching
    text_lower = text.lower()
    
    # Look for obvious category indicators
    if any(word in text_lower for word in ['electrolyte', 'sport', 'energy', 'drink']):
        result['categories'] = 'Sports/Energy Drink'
    elif any(word in text_lower for word in ['protein', 'supplement']):
        result['categories'] = 'Protein Supplement'
    elif any(word in text_lower for word in ['vitamin', 'mineral']):
        result['categories'] = 'Vitamin/Mineral Supplement'
    
    # Extract any recognizable words that could be product names
    words = text.split()
    potential_names = []
    
    for word in words:
        # Clean the word
        clean_word = re.sub(r'[^\w]', '', word)
        if (len(clean_word) >= 3 and 
            clean_word.isalpha() and
            clean_word.lower() not in ['the', 'and', 'for', 'with', 'per', 'serving']):
            potential_names.append(clean_word.title())
        
        if len(potential_names) >= 3:
            break
    
    if potential_names:
        result['product_name'] = ' '.join(potential_names[:3])
    
    # Look for any numeric values that might be nutrition info
    nutrition_patterns = [
        r'(\d+)\s*g(?:ram)?s?',  # grams
        r'(\d+)\s*mg',           # milligrams  
        r'(\d+)\s*cal',          # calories
        r'(\d+)\s*%',            # percentages
    ]
    
    nutrition_found = {}
    for pattern in nutrition_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            unit = pattern.split('\\')[1].strip('s*()[]?')
            for i, match in enumerate(matches[:3]):  # Limit to first 3 matches
                key = f"Value_{i+1}" if unit == 'd+' else f"Amount_{unit}"
                nutrition_found[key] = f"{match}{unit}"
    
    result['nutriments'] = nutrition_found
    
    # Look for brand-like patterns (all caps words)
    brand_matches = re.findall(r'\b[A-Z]{2,}\b', text)
    if brand_matches:
        # Filter out common non-brand all-caps words
        filtered_brands = [b for b in brand_matches 
                          if b not in ['THE', 'AND', 'FOR', 'WITH', 'PER', 'NET', 'WT']]
        if filtered_brands:
            result['brands'] = filtered_brands[0]
    
    return result

def clean_text(text, apply_spellcheck=False):
    """Much more aggressive cleaning for extremely noisy OCR"""
    if not text:
        return ""

    # First pass: Remove obvious noise patterns
    text = re.sub(r'[^\w\s.,;:()\-/&%]', ' ', text)
    text = re.sub(r'\b[a-zA-Z]{1}\b', ' ', text)  # Remove single letters
    text = re.sub(r'\b\d{1}\b', ' ', text)        # Remove single digits
    text = re.sub(r'\b[a-zA-Z]{2}\b(?![a-zA-Z])', ' ', text)  # Remove most 2-letter fragments
    
    # Remove fragments with mixed case patterns that look like OCR noise
    text = re.sub(r'\b[a-z]*[A-Z]+[a-z]*[A-Z]+[a-z]*\b', ' ', text)
    
    # Remove words with excessive consonants (OCR noise pattern)
    words = text.split()
    clean_words = []
    
    for word in words:
        if len(word) < 2:
            continue
            
        # Skip words with too many consonants vs vowels
        vowels = sum(1 for c in word.lower() if c in 'aeiou')
        consonants = sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')
        
        if len(word) > 4 and vowels == 0:  # No vowels in long word
            continue
        if len(word) > 6 and consonants > vowels * 3:  # Too many consonants
            continue
            
        # Skip words that are mostly repeated characters
        if len(set(word.lower())) < len(word) // 3 and len(word) > 3:
            continue
            
        clean_words.append(word)
    
    text = ' '.join(clean_words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_meaningful_phrases(text):
    """Extract potentially meaningful phrases from noisy text"""
    if not text:
        return []
    
    # Split into sentences/phrases
    phrases = re.split(r'[.!?]+', text)
    meaningful_phrases = []
    
    for phrase in phrases:
        phrase = phrase.strip()
        if len(phrase) < 10:  # Too short
            continue
            
        words = phrase.split()
        if len(words) < 3:  # Too few words
            continue
            
        # Count meaningful vs noise words
        meaningful_words = 0
        for word in words:
            if (len(word) >= 3 and 
                word.isalpha() and 
                sum(1 for c in word.lower() if c in 'aeiou') >= 1):
                meaningful_words += 1
        
        # Keep phrases with at least 60% meaningful words
        if meaningful_words / len(words) >= 0.6:
            meaningful_phrases.append(phrase)
    
    return meaningful_phrases

def advanced_parse_ocr_text(text):
    """
    Use the specialized parser for poor OCR results
    """
    return smart_parse_poor_ocr(text)


def ensure_dpi(img_path, min_dpi=200):  # Increased from 150 to 200
    img = None
    try:
        img = Image.open(img_path)
        dpi = img.info.get('dpi', (72, 72))[0]
        
        if dpi < min_dpi:
            scale = min_dpi / dpi
            new_size = (int(img.width * scale), int(img.height * scale))
            resized_img = img.resize(new_size, Image.LANCZOS)
            
            temp_path = f"upsampled_dpi_temp_{int(time.time() * 1000)}.jpg"
            resized_img.save(temp_path, dpi=(min_dpi, min_dpi))
            
            # Close images properly
            resized_img.close()
            img.close()
            
            return temp_path
        else:
            img.close()
            return img_path
            
    except Exception as e:
        if img:
            img.close()
        logger.warning(f"DPI adjustment failed: {e}")
        return img_path
    finally:
        if img:
            img.close()

# Replace the extract_product_info function in your ocr_module.py

def extract_product_info(image_path):
    """
    Main function to extract all product information with enhanced OCR and error handling
    """
    logger.info(f"Processing image: {image_path}")
    
    # Default return structure
    default_result = {
        'product_name': 'N/A',
        'ingredients_text': 'N/A',
        'nutriments': {},
        'brands': 'N/A',
        'categories': 'N/A',
        'barcode': None,
        'extracted_text': ''
    }

    try:
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            default_result['extracted_text'] = 'Error: Image file not found'
            return default_result

        # Ensure minimum DPI
        try:
            processed_image_path = ensure_dpi(image_path, min_dpi=200)
        except Exception as e:
            logger.warning(f"DPI adjustment failed: {e}")
            processed_image_path = image_path

        # Extract text using advanced preprocessing and multi-OCR
        try:
            final_text = extract_text_multiple_methods(processed_image_path)
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            default_result['extracted_text'] = f'Error during text extraction: {str(e)}'
            return default_result

        # Clean up temporary DPI file if created
        if processed_image_path != image_path:
            safe_remove_file(processed_image_path)

        if not final_text or not final_text.strip():
            logger.warning("No text extracted from image")
            default_result['extracted_text'] = 'No readable text found in image'
            return default_result

        # Parse structured info from the best OCR result
        try:
            parsed_info = advanced_parse_ocr_text(final_text)
            if not isinstance(parsed_info, dict):
                logger.error("Parsing returned invalid data structure")
                parsed_info = default_result.copy()
        except Exception as e:
            logger.error(f"Text parsing failed: {e}")
            parsed_info = default_result.copy()
            parsed_info['extracted_text'] = final_text

        # Extract barcode separately
        try:
            barcode = extract_barcode(image_path)
            parsed_info['barcode'] = barcode
        except Exception as e:
            logger.warning(f"Barcode extraction failed: {e}")
            parsed_info['barcode'] = None

        # Store extracted raw text
        parsed_info['extracted_text'] = final_text

        # Ensure all required fields exist
        for key in default_result:
            if key not in parsed_info:
                parsed_info[key] = default_result[key]

        return parsed_info

    except Exception as e:
        logger.error(f"Critical error in extract_product_info: {e}")
        default_result['extracted_text'] = f'Critical error: {str(e)}'
        return default_result

def extract_barcode(image_path):
    """Enhanced barcode extraction with better preprocessing"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Try multiple preprocessing approaches for barcode
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # IMPROVED: Better preprocessing for barcode detection
    processed_images = [
        gray,
        cv2.GaussianBlur(gray, (3, 3), 0),
        cv2.bilateralFilter(gray, 9, 75, 75),  # Better edge preservation
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ]
    
    # Try different regions of the image
    h, w = gray.shape
    regions = [
        gray,  # Full image
        gray[int(h*0.7):, :],  # Bottom 30%
        gray[:int(h*0.3), :],  # Top 30%
        gray[:, int(w*0.8):],  # Right 20%
    ]
    
    all_candidates = []
    
    for region in regions:
        for processed in processed_images:
            try:
                # OpenCV barcode detector
                detector = cv2.barcode_BarcodeDetector()
                result = detector.detectAndDecode(processed)
                
                if len(result) >= 3 and result[0] and len(str(result[0])) >= 8:
                    barcode_val = str(result[0])
                    if barcode_val.isdigit() and 8 <= len(barcode_val) <= 13:
                        all_candidates.append(barcode_val)
            except Exception:
                continue
    
    # Return most common valid barcode
    if all_candidates:
        from collections import Counter
        return Counter(all_candidates).most_common(1)[0][0]
    
    # Fallback: OCR-based extraction with better patterns
    try:
        config = '--psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(gray, config=config)
        
        # Look for common barcode patterns
        patterns = [
            r'\b(\d{13})\b',  # EAN-13
            r'\b(\d{12})\b',  # UPC-A
            r'\b(\d{8})\b',   # EAN-8
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
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