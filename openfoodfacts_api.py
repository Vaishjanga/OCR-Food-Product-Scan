import requests

def get_product_by_barcode(barcode):
    """
    Query Open Food Facts API by barcode.
    Returns product details as a dict if found, else None.
    """
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    resp = requests.get(url)
    if resp.status_code == 200 and resp.json().get('status') == 1:
        return resp.json()['product']
    return None

def get_product_by_name(name):
    """
    Query Open Food Facts API by product name (search).
    Returns the first matching product as a dict if found, else None.
    """
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        'search_terms': name,
        'search_simple': 1,
        'action': 'process',
        'json': 1,
    }
    resp = requests.get(url, params=params)
    products = resp.json().get('products', [])
    return products[0] if products else None