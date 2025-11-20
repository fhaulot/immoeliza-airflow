"""
Script principal pour scraper les propriétés Immovlan et les transformer 
au format attendu (compatible avec cleaned_data.csv).
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import uuid
from fake_headers import Headers
import random
import time
from typing import Dict, Optional, List
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_headers() -> dict:
    """Generate randomized headers to avoid blocking."""
    browsers = ["chrome", "firefox", "opera", "safari", "edge"]
    os_choices = ["win", "mac", "linux"]
    
    headers = Headers(
        browser=random.choice(browsers),
        os=random.choice(os_choices),
        headers=True)
    return headers.generate()

def extract_postal_code_from_url(url: str) -> Optional[int]:
    """Extract postal code from Immovlan URL."""
    if '/detail/' in url:
        parts = url.split('/')
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
    return None

def extract_locality_from_url(url: str) -> Optional[str]:
    """Extract locality from Immovlan URL."""
    try:
        parts = url.split('/')
        if len(parts) >= 8:
            locality = parts[7].replace('-', ' ').title()
            return locality
    except:
        pass
    return None

def get_province_from_postal_code(postal_code: int) -> str:
    """Map postal code to Belgian province."""
    if 1000 <= postal_code <= 1299:
        return "Brussels"
    elif 1300 <= postal_code <= 1499:
        return "Brabant Wallon"
    elif 1500 <= postal_code <= 1999:
        return "Vlaams-Brabant"
    elif 2000 <= postal_code <= 2999:
        return "Antwerp"
    elif 3000 <= postal_code <= 3499:
        return "Vlaams-Brabant"
    elif 3500 <= postal_code <= 3999:
        return "Limburg"
    elif 4000 <= postal_code <= 4999:
        return "Liège"
    elif 5000 <= postal_code <= 5999:
        return "Namur"
    elif 6000 <= postal_code <= 6999:
        return "Hainaut"
    elif 7000 <= postal_code <= 7999:
        return "Hainaut"
    elif 8000 <= postal_code <= 8999:
        return "West-Flanders"
    elif 9000 <= postal_code <= 9999:
        return "East-Flanders"
    else:
        return "Unknown"

def get_region_from_province(province: str) -> str:
    """Map province to region."""
    wallonia = ["Brabant Wallon", "Liège", "Namur", "Hainaut", "Luxembourg"]
    flanders = ["Antwerp", "Vlaams-Brabant", "Limburg", "West-Flanders", "East-Flanders"]
    
    if province in wallonia:
        return "Wallonia"
    elif province in flanders:
        return "Flanders"
    elif province == "Brussels":
        return "Brussels"
    else:
        return "Unknown"

def extract_property_type_from_url(url: str) -> tuple:
    """Extract property type and subtype from URL."""
    type_mapping = {
        'residence': ('HOUSE', 'HOUSE'),
        'house': ('HOUSE', 'HOUSE'), 
        'villa': ('HOUSE', 'VILLA'),
        'apartment': ('APARTMENT', 'APARTMENT'),
        'flat': ('APARTMENT', 'FLAT'),
        'duplex': ('APARTMENT', 'DUPLEX'),
        'penthouse': ('APARTMENT', 'PENTHOUSE'),
        'studio': ('APARTMENT', 'FLAT_STUDIO')
    }
    
    url_lower = url.lower()
    for key, (prop_type, subtype) in type_mapping.items():
        if f'/{key}/' in url_lower:
            return prop_type, subtype
    
    return 'HOUSE', 'HOUSE'  # Default

def extract_property_details(url: str) -> Optional[Dict]:
    """
    Extract comprehensive property details from an Immovlan property page.
    Returns data in the format expected by cleaned_data.csv.
    """
    try:
        req = requests.get(url, headers=get_headers())
        
        if req.status_code != 200:
            logger.warning(f"Failed to fetch {url}. Status: {req.status_code}")
            return None
            
        soup = BeautifulSoup(req.content, 'html.parser')
        
        # Initialize property data with required structure
        property_data = {
            'id': float(uuid.uuid4().int % (10**8)),  # Generate unique ID
            'url': url,
            'type': None,
            'subtype': None,
            'bedroomCount': None,
            'province': None,
            'locality': None,
            'postCode': None,
            'habitableSurface': None,
            'buildingCondition': None,
            'hasGarden': 0,
            'gardenSurface': 0.0,
            'hasTerrace': 0,
            'epcScore': None,
            'price': None,
            'hasParking': 0,
            'MunicipalityCleanName': None,
            'region': None,
            'price_square_meter': None
        }
        
        # Extract basic info from URL
        property_data['postCode'] = extract_postal_code_from_url(url)
        property_data['locality'] = extract_locality_from_url(url)
        property_data['type'], property_data['subtype'] = extract_property_type_from_url(url)
        
        if property_data['postCode']:
            property_data['province'] = get_province_from_postal_code(property_data['postCode'])
            property_data['region'] = get_region_from_province(property_data['province'])
            property_data['MunicipalityCleanName'] = property_data['locality']
        
        # Extract data from cXenseParse meta tags (most reliable)
        meta_mappings = {
            'cXenseParse:rbf-immovlan-prix': 'price',
            'cXenseParse:rbf-immovlan-chambres': 'bedroomCount', 
            'cXenseParse:rbf-immovlan-type': 'property_type_meta',
            'cXenseParse:rbf-immovlan-peb': 'epc_score_meta',
            'cXenseParse:rbf-immovlan-jardin': 'hasGarden_meta',
            'cXenseParse:rbf-immovlan-garage': 'hasParking_meta'
        }
        
        for meta_name, field_name in meta_mappings.items():
            meta_tag = soup.find('meta', attrs={'name': meta_name})
            if meta_tag and meta_tag.get('content'):
                content = meta_tag['content']
                if field_name in ['price', 'bedroomCount']:
                    # Extract numeric value
                    numeric_match = re.search(r'(\d+(?:\.\d+)?)', content)
                    if numeric_match:
                        property_data[field_name.replace('_meta', '')] = float(numeric_match.group(1))
                else:
                    property_data[field_name] = content
        
        # Process garden info
        if property_data.get('hasGarden_meta'):
            property_data['hasGarden'] = 1 if property_data['hasGarden_meta'].lower() in ['oui', 'yes', '1'] else 0
        
        # Process parking info 
        if property_data.get('hasParking_meta'):
            parking_info = property_data['hasParking_meta']
            # Look for numbers in parking info
            parking_match = re.search(r'(\d+)', parking_info)
            if parking_match:
                property_data['hasParking'] = 1 if int(parking_match.group(1)) > 0 else 0
        
        # Clean EPC score
        if property_data.get('epc_score_meta'):
            epc = property_data['epc_score_meta']
            epc_match = re.search(r'([A-G])$', epc)
            if epc_match:
                property_data['epcScore'] = epc_match.group(1)
        
        # Extract surface from meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content']
            surface_match = re.search(r'Livable surface\s*(\d+)m²', desc)
            if surface_match:
                property_data['habitableSurface'] = float(surface_match.group(1))
        
        # Try to extract more details from general-info section
        general_info = soup.find('div', class_='general-info')
        if general_info:
            text = general_info.get_text()
            
            # Extract livable surface if not found
            if not property_data['habitableSurface']:
                surface_match = re.search(r'Livable surface\s*(\d+)\s*m²', text)
                if surface_match:
                    property_data['habitableSurface'] = float(surface_match.group(1))
            
            # Extract bedrooms if not found
            if not property_data['bedroomCount']:
                bedroom_match = re.search(r'Number of bedrooms\s*(\d+)', text)
                if bedroom_match:
                    property_data['bedroomCount'] = float(bedroom_match.group(1))
            
            # Extract garden surface
            garden_surface_match = re.search(r'Surface garden\s*(\d+)\s*m²', text)
            if garden_surface_match:
                property_data['gardenSurface'] = float(garden_surface_match.group(1))
                property_data['hasGarden'] = 1
            
            # Check for terrace
            if 'terrace' in text.lower() or 'TerraceYes' in text:
                property_data['hasTerrace'] = 1
                
            # Extract building condition
            conditions = ['AS_NEW', 'GOOD', 'JUST_RENOVATED', 'TO_RENOVATE', 'TO_BE_DONE_UP', 'TO_RESTORE']
            for condition in conditions:
                if condition.replace('_', ' ').lower() in text.lower():
                    property_data['buildingCondition'] = condition
                    break
        
        # Calculate price per square meter
        if property_data['price'] and property_data['habitableSurface']:
            property_data['price_square_meter'] = property_data['price'] / property_data['habitableSurface']
        
        # Clean up temporary meta fields
        keys_to_remove = [k for k in property_data.keys() if k.endswith('_meta')]
        for key in keys_to_remove:
            del property_data[key]
        
        # Ensure we have minimum required data
        required_fields = ['postCode', 'price']
        if all(property_data.get(field) is not None for field in required_fields):
            return property_data
        else:
            logger.warning(f"Missing required data for {url}: {[f for f in required_fields if property_data.get(f) is None]}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return None

def scrape_properties_batch(urls: List[str], start_idx: int = 0, batch_size: int = 100, output_file: str = None) -> pd.DataFrame:
    """
    Scrape a batch of properties and save incrementally.
    """
    end_idx = min(start_idx + batch_size, len(urls))
    batch_urls = urls[start_idx:end_idx]
    
    logger.info(f"Processing batch {start_idx//batch_size + 1}: properties {start_idx+1}-{end_idx} of {len(urls)}")
    
    properties_data = []
    
    for i, url in enumerate(batch_urls, start_idx + 1):
        logger.info(f"Scraping property {i}/{len(urls)}: {url}")
        
        property_data = extract_property_details(url)
        if property_data:
            properties_data.append(property_data)
            logger.info(f"  ✓ Success: Price={property_data.get('price', 'N/A')}€, "
                       f"Postal={property_data.get('postCode', 'N/A')}, "
                       f"Bedrooms={property_data.get('bedroomCount', 'N/A')}, "
                       f"Surface={property_data.get('habitableSurface', 'N/A')}m²")
        else:
            logger.warning(f"  ✗ Failed to extract data")
        
        # Random delay
        time.sleep(random.uniform(1, 3))
    
    if properties_data:
        df = pd.DataFrame(properties_data)
        
        # Save batch if output file specified
        if output_file:
            # Check if file exists to decide whether to write header
            file_exists = Path(output_file).exists()
            df.to_csv(output_file, mode='a', header=not file_exists, index=False, encoding='utf-8')
            logger.info(f"Batch saved to {output_file}: {len(properties_data)} properties")
        
        return df
    else:
        return pd.DataFrame()

def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls

def main():
    """Main function to scrape all properties."""
    # Configuration - use environment variables or defaults
    import os
    urls_file = os.getenv('URLS_FILE', '/app/data/immovlan_sales_urls.txt')
    output_file = os.getenv('OUTPUT_FILE', '/app/data/immovlan_scraped_data.csv')
    batch_size = int(os.getenv('BATCH_SIZE', '100'))
    test_mode = os.getenv('TEST_MODE', 'true').lower() == 'true'
    
    # Load URLs
    urls = load_urls_from_file(urls_file)
    logger.info(f"Loaded {len(urls)} URLs from {urls_file}")
    
    if test_mode:
        # Test with first 50 properties
        logger.info("=== RUNNING IN TEST MODE (first 50 properties) ===")
        urls = urls[:10]
        # Keep the output file from env variable for test mode too
    
    # Remove existing output file to start fresh
    if Path(output_file).exists():
        Path(output_file).unlink()
    
    # Process in batches
    all_data = []
    for start_idx in range(0, len(urls), batch_size):
        batch_df = scrape_properties_batch(urls, start_idx, batch_size, output_file)
        if not batch_df.empty:
            all_data.append(batch_df)
        
        # Progress report
        processed = min(start_idx + batch_size, len(urls))
        logger.info(f"Progress: {processed}/{len(urls)} URLs processed")
    
    # Combine all data and show summary
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\n=== SCRAPING COMPLETED ===")
        logger.info(f"Total properties scraped: {len(final_df)}")
        logger.info(f"Success rate: {len(final_df)}/{len(urls)} ({100*len(final_df)/len(urls):.1f}%)")
        logger.info(f"Data saved to: {output_file}")
        
        # Show sample and column info
        logger.info(f"\nColumns: {list(final_df.columns)}")
        logger.info(f"Sample data:")
        logger.info(final_df.head().to_string())
        
        # Show data quality stats
        logger.info(f"\nData quality:")
        logger.info(f"Properties with price: {final_df['price'].notna().sum()}")
        logger.info(f"Properties with surface: {final_df['habitableSurface'].notna().sum()}")
        logger.info(f"Properties with bedrooms: {final_df['bedroomCount'].notna().sum()}")
        logger.info(f"Properties with EPC: {final_df['epcScore'].notna().sum()}")
        
    else:
        logger.error("No properties were successfully scraped!")

if __name__ == "__main__":
    main()