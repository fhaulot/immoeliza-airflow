"""
Script to collect all sales property URLs from Immovlan by province and property type (English version).
- Uses Selenium (undetected_chromedriver) to accept cookies once.
- Scrapes multiple pages for each province and property type using requests and BeautifulSoup.
- Prints scraping progress and saves URLs to a file.
"""

import requests
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# List of provinces and property types in English
provinces = [
    "brussels", "vlaams-brabant", "brabant-wallon", "liege", "namur",
    "hainaut", "luxembourg", "east-flanders", "west-flanders", "limburg"
]
property_types = ["house", "apartment"]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.6099.129 Safari/537.36"
}

def accept_cookies():
    driver = uc.Chrome()
    driver.get("https://immovlan.be/en/real-estate?transactiontypes=for-sale,in-public-sale&propertytypes=house,apartment")
    time.sleep(8)
    try:
        cookie_button = driver.find_element(By.XPATH, '//*[@id="didomi-notice-agree-button"]')
        cookie_button.click()
        print("Cookies accepted.")
    except Exception as e:
        print("Cookie button not found or already accepted.")
    driver.quit()

def scrape_page(root_url, page_number, sales_url):
    url = f"{root_url}&page={page_number}&noindex=1"
    req = requests.get(url, headers=headers)
    if req.status_code == 200:
        soup = BeautifulSoup(req.text, 'html.parser')
        links = soup.find_all('a', href=True)
        for link in links:
            href = link["href"]
            if "/detail/" in href:
                full_url = "https://immovlan.be" + href if href.startswith("/") else href
                if full_url not in sales_url:
                    sales_url.append(full_url)
    else:
        print(f"Error while scraping page {page_number}. Status code = {req.status_code}")

if __name__ == "__main__":
    accept_cookies()  # Accept cookies once

    all_sales_urls = []

    for province in provinces:
        for prop_type in property_types:
            print(f"\nScraping {prop_type}s in {province}...")
            root_url = f"https://immovlan.be/en/real-estate?transactiontypes=for-sale,in-public-sale&propertytypes={prop_type}&provinces={province}"
            sales_url = []
            for page in range(1, 51):  # Adjust number of pages if needed
                scrape_page(root_url, page, sales_url)
                print(f"  {prop_type.capitalize()}s in {province}: Page {page} scraped. Total URLs collected: {len(sales_url)}")
                time.sleep(1)
            print(f"Finished {prop_type}s in {province}. Total URLs: {len(sales_url)}")
            all_sales_urls.extend(sales_url)

    print(f"\nScraping finished. Total unique URLs collected: {len(set(all_sales_urls))}")

    # Save URLs to a file
    with open("immovlan_sales_urls.txt", "w", encoding="utf-8") as f:
        for url in set(all_sales_urls):
            f.write(url + "\n")
    print("URLs saved to immovlan_sales_urls.txt")