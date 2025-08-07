OCR-Based Product Information Scanner

This project is an advanced OCR-based product information scanner that extracts text from product images and retrieves detailed product information from the Open Food Facts database. The system uses multiple OCR engines and sophisticated preprocessing techniques to handle even poor-quality images.

Features

1.Multi-Engine OCR: Uses both Tesseract and EasyOCR for maximum text extraction accuracy
2.Advanced Image Preprocessing: Multiple preprocessing strategies for handling low-quality images
3.Intelligent Text Parsing: Extracts product names, ingredients, nutrition facts, brands, and categories
4.Barcode Detection: Automatic barcode extraction and validation
5.Open Food Facts Integration**: Searches products by barcode and name
6.Web Interface: Clean Streamlit-based interface with debug information
7.Robust Error Handling: Comprehensive error handling for all components

Technical Stack
Python 3.7+
OCR Engines: Tesseract, EasyOCR
Image Processing: OpenCV, PIL
NLP: spaCy, TextBlob
API Integration: Open Food Facts API
Web Interface: Streamlit
Text Processing: RapidFuzz for fuzzy matching

Project Structure

app.py                  # Main Streamlit application
ocr_module.py          # OCR processing and text 
openfoodfacts_api.py   # Open Food Facts API integration
requirements.txt       # Python dependencies
README.md             # This file


Installation & Setup

Prerequisites
1. Install Tesseract OCR:
   - Windows: Download from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

2. Python 3.7 or higher

Step-by-Step Installation

1. Clone or download the project files

   git clone <https://github.com/Vaishjanga/OCR-Food-Product-Scan.git>
   cd ocr-product-scanner


2. Create a virtual environment (recommended)
 
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install Python dependencies

   pip install -r requirements.txt


4. Download spaCy English model
   python -m spacy download en_core_web_sm


5. Configure Tesseract path(Windows only)
   - Update the path in `ocr_module.py` if Tesseract is installed in a different location:

   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'


Usage

Running the Application

streamlit run app.py


The application will open in your web browser at `http://localhost:8501`

Using the Interface
1. Upload Image: Click "Upload a product image" and select a JPG, JPEG, or PNG file
2. Optional Barcode: Enter a barcode manually if known
3. Processing: The system will automatically:
   - Extract text using multiple OCR methods
   - Parse product information
   - Search Open Food Facts database
   - Display results
4. Debug Information: Expand the debug section to see raw OCR output and parsing details

Supported Image Formats
- JPG/JPEG
- PNG
- Handles various image qualities and resolutions
- Automatic DPI upscaling for better OCR results

How It Works

1. Image Preprocessing
- DPI Enhancement: Automatically upscales         
    low-resolution images
- Multiple Preprocessing Strategies:
    High contrast enhancement using CLAHE
    Gaussian blur with Otsu thresholding
    Adaptive thresholding
    Morphological operations (opening, erosion, dilation)

2. OCR Text Extraction
- EasyOCR: Better for handwritten text and complex layouts
- Tesseract: Multiple PSM modes for different text patterns
- Quality Scoring: Automatically selects best OCR result based on readability metrics

3. Information Parsing
- Fuzzy Keyword Matching: Identifies ingredients, nutrition facts, brands
- Named Entity Recognition: Uses spaCy for intelligent text analysis
- Pattern Recognition: Extracts nutrition values, product categories

4. Database Integration
- Barcode Search: Primary search method using extracted or manual barcodes
- Name Search: Fallback search using extracted product names
- Data Formatting: Standardizes nutrition information display

API Integration

Open Food Facts API
The system integrates with the Open Food Facts database:
- Barcode Lookup: `https://world.openfoodfacts.org/api/v0/product/{barcode}.json`
- Name Search: `https://world.openfoodfacts.org/cgi/search.pl`
- Timeout Handling: 10-15 second timeouts with error recovery
- Rate Limiting: Respectful API usage

Error Handling

The system includes comprehensive error handling for:
- Image Loading Errors: Invalid file formats, corrupted images
- OCR Failures: Fallback to alternative OCR methods
- API Timeouts: Graceful degradation to OCR-only results
- Network Issues: Offline mode with extracted data
- File Cleanup: Automatic temporary file removal

Performance Tips
- Image Quality: Higher resolution images generally yield better results
- Lighting: Good contrast between text and background
- Focus: Sharp, clear text improves OCR accuracy
- Orientation: Ensure text is right-side up

Sample Test Cases

Good Test Images
- Clear product labels with visible text
- Nutrition fact panels
- Ingredient lists
- Barcodes on product packaging

Challenging Cases
- Curved or wrinkled labels
- Poor lighting conditions
- Small text or low resolution
- Handwritten elements

Technical Approach

OCR Strategy
1. Multi-Method Approach: Combines multiple OCR engines for best results
2. Adaptive Preprocessing: Different strategies for different image types
3. Quality Assessment: Automatic selection of best OCR output
4. Confidence Scoring: Filters low-confidence results

 Text Processing
1. Noise Reduction: Removes OCR artifacts and invalid characters
2. Fuzzy Matching: Handles OCR errors in keyword recognition
3. Context Analysis: Uses surrounding text for better parsing
4. Structured Extraction: Organized extraction of different data types

Future Enhancements

- Multi-language Support: Extend beyond English
- Batch Processing: Handle multiple images at once
- Mobile Integration: Camera capture functionality
- Custom Product Database: Local database for faster lookups
- Machine Learning: Improve parsing accuracy with ML models
