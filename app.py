import streamlit as st
from ocr_module import extract_product_info
from openfoodfacts_api import get_product_by_barcode, get_product_by_name
import os

# Set Streamlit layout and title
st.set_page_config(page_title="OCR Product Scanner", layout="wide")
st.title("📦 OCR-Based Product Information Scanner")

# OCR accuracy highlights
st.info("""
**🔍 OCR Accuracy Improvements:**
- ✅ Advanced image preprocessing: denoising, CLAHE, sharpening, gamma correction
- ✅ Adaptive DPI normalization for OCR-friendly scaling
- ✅ Multi-engine OCR: Tesseract (with multiple configs) + EasyOCR
- ✅ Text cleaning and spell-checking (TextBlob)
- ✅ Fuzzy keyword detection + NLP (spaCy) for structured data
""")

# Upload image and/or barcode
uploaded_file = st.file_uploader("📤 Upload a product image", type=["jpg", "jpeg", "png"])
manual_barcode = st.text_input("✏️ Enter barcode manually (optional):")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save image temporarily
    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Step 1: Extract product info via OCR pipeline
    with st.spinner("🔍 Extracting product information..."):
        extracted = extract_product_info(temp_path)

    # Step 2: Use manual barcode if provided
    if manual_barcode.strip():
        extracted["barcode"] = manual_barcode.strip()
        st.info(f"Using manually entered barcode: `{extracted['barcode']}`")

    # Step 3: Try Open Food Facts API
    product = None
    if extracted.get("barcode"):
        with st.spinner("🔎 Searching Open Food Facts by barcode..."):
            product = get_product_by_barcode(extracted["barcode"])

    if not product and extracted.get("product_name"):
        with st.spinner("🔎 Searching Open Food Facts by product name..."):
            product = get_product_by_name(extracted["product_name"])

    # Step 4: Fallback to OCR data if no match
    if not product:
        st.warning("⚠️ No product found in Open Food Facts. Displaying extracted OCR data.")
        product = extracted

    # Step 6: Display extracted or fetched product info
    st.subheader("📦 Product Details")
    st.markdown(f"**🛍️ Product Name:** {product.get('product_name', 'N/A')}")
    st.markdown(f"**🏷️ Brand:** {product.get('brands', 'N/A')}")
    st.markdown(f"**📂 Category:** {product.get('categories', 'N/A')}")
    st.markdown(f"**🧾 Ingredients:** {product.get('ingredients_text', 'N/A')}")

    st.markdown("**🍽️ Nutrition Facts:**")
    nutriments = product.get("nutriments", {})
    if nutriments:
        st.table([[key, value] for key, value in nutriments.items()])
    else:
        st.write("N/A")

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    st.success("✅ Product information extracted successfully!")
