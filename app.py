import streamlit as st
from ocr_module import extract_product_info
from openfoodfacts_api import get_product_by_barcode, get_product_by_name
import os
import time

# Set Streamlit layout and title
st.set_page_config(page_title="OCR Product Scanner", layout="wide")
st.title("OCR-Based Product Information Scanner")

# Upload image and/or barcode
uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
manual_barcode = st.text_input(" Enter barcode manually (optional):")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save image temporarily with unique filename
    import time
    timestamp = str(int(time.time() * 1000))
    temp_path = f"temp_uploaded_image_{timestamp}.jpg"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Step 1: Extract product info via OCR pipeline
        with st.spinner("Extracting product information..."):
            try:
                extracted = extract_product_info(temp_path)
                # Ensure extracted is a dictionary
                if not isinstance(extracted, dict):
                    st.error("OCR extraction failed - invalid return type")
                    extracted = {
                        'product_name': 'N/A',
                        'ingredients_text': 'N/A',
                        'nutriments': {},
                        'brands': 'N/A',
                        'categories': 'N/A',
                        'barcode': None,
                        'extracted_text': 'OCR processing failed'
                    }
            except Exception as ocr_error:
                st.error(f"OCR extraction failed: {str(ocr_error)}")
                extracted = {
                    'product_name': 'N/A',
                    'ingredients_text': 'N/A',
                    'nutriments': {},
                    'brands': 'N/A',
                    'categories': 'N/A',
                    'barcode': None,
                    'extracted_text': f'Error: {str(ocr_error)}'
                }

        with st.expander("Debug Information (Click to expand)"):
            st.write("Raw Extracted Text:")
            st.text_area("OCR Output", value=extracted.get('extracted_text', 'No text extracted'), height=150)
            st.write("**Parsed Data:**")
            st.json(extracted)
            
            # Show image properties
            from PIL import Image
            try:
                pil_img = Image.open(temp_path)
                st.write(f"**Image Info:** {pil_img.size} pixels, Mode: {pil_img.mode}")
                if hasattr(pil_img, 'info') and 'dpi' in pil_img.info:
                    st.write(f"**DPI:** {pil_img.info['dpi']}")
                pil_img.close()  # Explicitly close the image
            except Exception as e:
                st.write(f"Could not read image info: {e}")

        # Step 2: Use manual barcode if provided
        if manual_barcode.strip():
            extracted["barcode"] = manual_barcode.strip()
            st.info(f"Using manually entered barcode: `{extracted['barcode']}`")

        # Step 3: Try Open Food Facts API
        product = None
        api_error = None
        
        if extracted.get("barcode"):
            with st.spinner("Searching Open Food Facts by barcode..."):
                try:
                    product = get_product_by_barcode(extracted["barcode"])
                except Exception as e:
                    api_error = f"Barcode search failed: {str(e)}"
                    st.warning(f"{api_error}")

        if not product and extracted.get("product_name") and extracted.get("product_name") != "N/A":
            with st.spinner(" Searching Open Food Facts by product name..."):
                try:
                    product = get_product_by_name(extracted["product_name"])
                except Exception as e:
                    api_error = f"Name search failed: {str(e)}"
                    st.warning(f" {api_error}")

        # Step 4: Fallback to OCR data if no match
        if not product:
            st.warning(" No product found in Open Food Facts. Displaying extracted OCR data.")
            product = extracted

        # Step 5: Display extracted or fetched product info
        st.subheader("Product Details")
        st.markdown(f" Product Name: {product.get('product_name', 'N/A')}")
        st.markdown(f" Brand: {product.get('brands', 'N/A')}")
        st.markdown(f" Category: {product.get('categories', 'N/A')}")
        st.markdown(f" Ingredients: {product.get('ingredients_text', 'N/A')}")

        st.markdown(" Nutrition Facts:")
        nutriments = product.get("nutriments", {})
        if nutriments:
            # Convert to proper format for streamlit table
            import pandas as pd
            nutrition_data = []
            for key, value in nutriments.items():
                # Ensure both key and value are strings
                nutrition_data.append([str(key), str(value)])
            
            if nutrition_data:
                df = pd.DataFrame(nutrition_data, columns=["Nutrient", "Value"])
                st.dataframe(df, use_container_width=True)
            else:
                st.write("N/A")
        else:
            st.write("N/A")

        st.success(" Product information extracted successfully!")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    
    finally:
        # Cleanup temp file with better error handling
        try:
            if os.path.exists(temp_path):
                # Force close any file handles
                import gc
                gc.collect()
                
                # Try multiple times with delay
                for attempt in range(3):
                    try:
                        os.remove(temp_path)
                        break
                    except PermissionError:
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(0.5)
                        else:
                            # If still can't delete, rename it for later cleanup
                            try:
                                os.rename(temp_path, f"cleanup_{timestamp}.jpg")
                            except:
                                pass
        except Exception:
            pass  
