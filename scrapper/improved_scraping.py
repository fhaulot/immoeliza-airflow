"""
Script amélioré pour scraper les détails des propriétés depuis Immovlan.
Extrait le code postal, prix, nombre de chambres, et surface habitable.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
from fake_headers import Headers
import random
import time
from typing import Dict, Optional, List

def get_headers() -> dict:
    """Generate randomized headers to avoid blocking."""
    browsers = ["chrome", "firefox", "opera", "safari", "edge"]
    os_choices = ["win", "mac", "linux"]
    
    headers = Headers(
        browser=random.choice(browsers),
        os=random.choice(os_choices),
        headers=True)
    return headers.generate()

def extract_property_details(url: str) -> Optional[Dict]:
    """
    Extract key property details from an Immovlan property page.
    
    Returns:
        Dict with postal_code, price, bedrooms, livable_surface, property_type, epc_score
    """
    try:
        req = requests.get(url, headers=get_headers())
        
        if req.status_code != 200:
            print(f"Failed to fetch {url}. Status: {req.status_code}")
            return None
            
        soup = BeautifulSoup(req.content, 'html.parser')
        
        property_data = {'url': url}
        
        # Extract postal code from URL
        if '/detail/' in url:
            parts = url.split('/')
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    property_data['postal_code'] = int(part)
                    break
        
        # Extract data from cXenseParse meta tags (most reliable method)
        meta_mappings = {
            'cXenseParse:rbf-immovlan-prix': 'price',
            'cXenseParse:rbf-immovlan-chambres': 'bedrooms', 
            'cXenseParse:rbf-immovlan-type': 'property_type',
            'cXenseParse:rbf-immovlan-peb': 'epc_score'
        }
        
        for meta_name, field_name in meta_mappings.items():
            meta_tag = soup.find('meta', attrs={'name': meta_name})
            if meta_tag and meta_tag.get('content'):
                content = meta_tag['content']
                if field_name in ['price', 'bedrooms']:
                    # Try to extract numeric value
                    numeric_match = re.search(r'(\\d+(?:\\.\\d+)?)', content)
                    if numeric_match:
                        property_data[field_name] = float(numeric_match.group(1))
                else:
                    property_data[field_name] = content
        
        # Extract livable surface from meta description or page content
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content']
            # Look for patterns like "Livable surface 100m²" or "100m²"
            surface_match = re.search(r'Livable surface\\s*(\\d+)m²', desc)
            if surface_match:
                property_data['livable_surface'] = int(surface_match.group(1))
        
        # Try to extract livable surface from the general info section if not found in meta
        if 'livable_surface' not in property_data:
            general_info = soup.find('div', class_='general-info')
            if general_info:
                text = general_info.get_text()
                surface_match = re.search(r'Livable surface\\s*(\\d+)\\s*m²', text)
                if surface_match:
                    property_data['livable_surface'] = int(surface_match.group(1))
        
        # Try to extract from JSON-LD structured data as backup
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                if script.string:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        # Look for price in structured data
                        if 'price' in data and 'price' not in property_data:
                            if isinstance(data['price'], (int, float)):
                                property_data['price'] = float(data['price'])
                        
                        # Look for floorSize or similar
                        if 'floorSize' in data and 'livable_surface' not in property_data:
                            if isinstance(data['floorSize'], (int, float)):
                                property_data['livable_surface'] = float(data['floorSize'])
                        
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        
        # Clean up EPC score if present
        if 'epc_score' in property_data:
            epc = property_data['epc_score']
            # Extract just the letter (A, B, C, etc.) from strings like "WalloniaG"
            epc_match = re.search(r'([A-G])$', epc)
            if epc_match:
                property_data['epc_score'] = epc_match.group(1)
        
        # Ensure we have the minimum required data
        required_fields = ['postal_code', 'price']
        if all(field in property_data for field in required_fields):
            return property_data
        else:
            print(f"Missing required data for {url}: {[f for f in required_fields if f not in property_data]}")
            return None
            
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def scrape_properties_from_urls(urls: List[str], max_properties: int = None) -> pd.DataFrame:
    """
    Scrape property details from a list of URLs.
    
    Args:
        urls: List of property URLs
        max_properties: Maximum number of properties to scrape (for testing)
    
    Returns:
        DataFrame with scraped property data
    """
    properties_data = []
    
    # Limit for testing if specified
    if max_properties:
        urls = urls[:max_properties]
    
    print(f"Starting to scrape {len(urls)} properties...")
    
    for i, url in enumerate(urls, 1):
        print(f"Scraping property {i}/{len(urls)}: {url}")
        
        property_data = extract_property_details(url)
        if property_data:
            properties_data.append(property_data)
            print(f"  ✓ Extracted: Price={property_data.get('price', 'N/A')}€, "
                  f"Postal={property_data.get('postal_code', 'N/A')}, "
                  f"Bedrooms={property_data.get('bedrooms', 'N/A')}, "
                  f"Surface={property_data.get('livable_surface', 'N/A')}m²")
        else:
            print(f"  ✗ Failed to extract data")
        
        # Random delay to be respectful to the server
        time.sleep(random.uniform(1, 3))
        
        # Progress update every 50 properties
        if i % 50 == 0:
            print(f"Progress: {i}/{len(urls)} properties scraped ({len(properties_data)} successful)")
    
    print(f"\\nScraping completed: {len(properties_data)} properties successfully scraped")
    
    if properties_data:
        df = pd.DataFrame(properties_data)
        return df
    else:
        return pd.DataFrame()

def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

if __name__ == "__main__":
    # Load URLs from file
    urls_file = '/home/floriane/GitHub/immoeliza-airflow/immovlan_sales_urls.txt'
    urls = load_urls_from_file(urls_file)
    
    print(f"Loaded {len(urls)} URLs from {urls_file}")
    
    # Test with first 10 properties
    print("\\n=== TESTING WITH FIRST 10 PROPERTIES ===")
    test_df = scrape_properties_from_urls(urls, max_properties=10)
    
    if not test_df.empty:
        print("\\n=== SAMPLE RESULTS ===")
        print(test_df.head())
        print(f"\\nColumns: {list(test_df.columns)}")
        print(f"Shape: {test_df.shape}")
        
        # Save test results
        test_df.to_csv('/home/floriane/GitHub/immoeliza-airflow/test_scraped_properties.csv', 
                      index=False, encoding='utf-8')
        print("\\nTest results saved to test_scraped_properties.csv")
    else:
        print("No data was successfully scraped.")