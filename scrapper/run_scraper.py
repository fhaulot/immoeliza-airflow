"""
Complete scraper orchestrator: URLs collection + Property details scraping
This script runs both phases of the scraping process.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_url_scraper():
    """Phase 1: Scrape property URLs from Immovlan"""
    logger.info("="*80)
    logger.info("PHASE 1: SCRAPING PROPERTY URLs")
    logger.info("="*80)
    
    # Import and run immovlan.py
    sys.path.insert(0, str(Path(__file__).parent))
    import immovlan
    
    # Override the output file path to use environment variable
    urls_output = os.getenv('URLS_FILE', '/app/data/immovlan_sales_urls.txt')
    
    logger.info("Starting URL collection from Immovlan...")
    
    # Accept cookies once
    immovlan.accept_cookies()
    
    all_sales_urls = []
    
    for province in immovlan.provinces:
        for prop_type in immovlan.property_types:
            logger.info(f"Scraping {prop_type}s in {province}...")
            root_url = f"https://immovlan.be/en/real-estate?transactiontypes=for-sale,in-public-sale&propertytypes={prop_type}&provinces={province}"
            sales_url = []
            
            # In test mode, only scrape 2 pages per province/type
            test_mode = os.getenv('TEST_MODE', 'true').lower() == 'true'
            max_pages = 2 if test_mode else 500  # 500 pages = all properties (~13000 URLs)
            
            for page in range(1, max_pages + 1):
                immovlan.scrape_page(root_url, page, sales_url)
                logger.info(f"  {prop_type.capitalize()}s in {province}: Page {page} scraped. Total URLs: {len(sales_url)}")
                immovlan.time.sleep(0.5)  # Reduced from 1s to 0.5s
            
            logger.info(f"Finished {prop_type}s in {province}. Total URLs: {len(sales_url)}")
            all_sales_urls.extend(sales_url)
    
    # Remove duplicates and save
    unique_urls = set(all_sales_urls)
    logger.info(f"Scraping finished. Total unique URLs collected: {len(unique_urls)}")
    
    # Save URLs to file
    with open(urls_output, "w", encoding="utf-8") as f:
        for url in unique_urls:
            f.write(url + "\n")
    
    logger.info(f"URLs saved to {urls_output}")
    return len(unique_urls)

def run_property_scraper():
    """Phase 2: Scrape property details"""
    logger.info("="*80)
    logger.info("PHASE 2: SCRAPING PROPERTY DETAILS")
    logger.info("="*80)
    
    # Import and run main_scraper.py
    from main_scraper import main
    
    logger.info("Starting property details scraping...")
    main()

def main():
    """Run complete scraping pipeline"""
    logger.info("="*80)
    logger.info("COMPLETE SCRAPER PIPELINE")
    logger.info("="*80)
    
    test_mode = os.getenv('TEST_MODE', 'true').lower() == 'true'
    if test_mode:
        logger.info("Running in TEST MODE (limited URLs and properties)")
    else:
        logger.info("Running in FULL MODE (all URLs and properties)")
    
    try:
        # Phase 1: Collect URLs
        url_count = run_url_scraper()
        logger.info(f"✓ Phase 1 complete: {url_count} URLs collected")
        
        # Phase 2: Scrape property details
        run_property_scraper()
        logger.info("✓ Phase 2 complete: Property details scraped")
        
        logger.info("="*80)
        logger.info("SCRAPER PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in scraper pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
