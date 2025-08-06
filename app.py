import streamlit as st
from ocr_module import extract_text, extract_barcode, parse_ocr_text
from openfoodfacts_api import get_product_by_barcode, get_product_by_name
import os
import re


st.title("OCR-Based Product Information Scanner")

# Add information about OCR improvements
st.info("""
**OCR Accuracy Improvements:**
- Advanced image preprocessing (denoising, contrast enhancement, morphological operations)
- Multiple OCR engines (Tesseract + EasyOCR) with combined results
- Text cleaning and error correction
- Improved pattern matching for product information extraction
""")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
manual_barcode = st.text_input("Enter barcode manually (optional):")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Extract text using improved OCR
    with st.spinner("Extracting text from image..."):
        text = extract_text(temp_path, method='combined')
    
    # Extract barcode
    with st.spinner("Extracting barcode..."):
        barcode = extract_barcode(temp_path)
    
    st.write("**Extracted Text:**", text)
    st.write("**Extracted Barcode:**", barcode if barcode else "Not found")
    
    # Use manual barcode if provided
    if manual_barcode.strip():
        barcode = manual_barcode.strip()
        st.info(f"Using manually entered barcode: {barcode}")
    
    product = None
    if barcode:
        with st.spinner("Searching Open Food Facts database..."):
            product = get_product_by_barcode(barcode)
    
    if not product and text:
        with st.spinner("Searching by product name..."):
            product = get_product_by_name(text)
    
    if not product:
        with st.spinner("Parsing information from OCR text..."):
            # Fallback: parse from OCR text
            product = parse_ocr_text(text)
    
    # Always display in the specified format
    st.subheader("Product Details")
    st.markdown(f"**Product name:** {product.get('product_name', 'N/A')}")
    st.markdown(f"**Ingredients:** {product.get('ingredients_text', 'N/A')}")
    st.markdown(f"**Brand:** {product.get('brands', 'N/A')}")
    st.markdown(f"**Product categories:** {product.get('categories', 'N/A')}")
    
    nutriments = product.get('nutriments', {})
    st.markdown("**Nutrition facts:**")
    if nutriments:
        # Create a more readable table
        nut_data = []
        for key, value in nutriments.items():
            nut_data.append([key, value])
        st.table(nut_data)
    else:
        st.write("N/A")
    
    os.remove(temp_path)