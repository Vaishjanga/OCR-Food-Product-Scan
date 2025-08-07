# Replace your entire openfoodfacts_api.py with this:

import requests
import json
import logging

logger = logging.getLogger(__name__)

def get_product_by_barcode(barcode):
    """
    Query Open Food Facts API by barcode.
    Returns product details as a dict if found, else None.
    """
    if not barcode or not str(barcode).strip():
        return None
        
    try:
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        resp = requests.get(url, timeout=10)
        
        if resp.status_code != 200:
            logger.warning(f"API returned status {resp.status_code} for barcode {barcode}")
            return None
            
        # Check if response is valid JSON
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for barcode {barcode}: {e}")
            return None
            
        if data.get('status') == 1 and 'product' in data:
            return data['product']
        else:
            logger.info(f"No product found for barcode {barcode}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching barcode {barcode}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for barcode {barcode}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching barcode {barcode}: {e}")
        return None

def get_product_by_name(name):
    """
    Query Open Food Facts API by product name (search).
    Returns the first matching product as a dict if found, else None.
    """
    if not name or not str(name).strip():
        return None
        
    try:
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            'search_terms': str(name).strip(),
            'search_simple': 1,
            'action': 'process',
            'json': 1,
        }
        
        resp = requests.get(url, params=params, timeout=15)
        
        if resp.status_code != 200:
            logger.warning(f"API returned status {resp.status_code} for search '{name}'")
            return None
            
        # Check if response is valid JSON
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for search '{name}': {e}")
            return None
            
        products = data.get('products', [])
        if products and len(products) > 0:
            return products[0]
        else:
            logger.info(f"No products found for search '{name}'")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while searching for '{name}'")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for search '{name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error searching for '{name}': {e}")
        return None