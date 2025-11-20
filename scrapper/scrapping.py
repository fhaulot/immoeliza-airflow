import requests
from bs4 import BeautifulSoup
import pandas as pd 
from fake_headers import Headers
import random
import time
from immovlan import sales_url

""" Importing modules :
Requests : to get the url
BeautifulSoup : to parse the html code
CSV : to get the data in a nice csv file
Pandas : to get the data from python to a structured sheet
Fake-headers : to fake a browser and pass the blocking of the website
Random : to get a different header each time
Time : to take between each request and don't charge too much the website 
sales_url : to get the urls list we got in the immovlan.py file """

def get_headers() -> dict:
    """
    Generate randomized, realistic HTTP headers to reduce request blocking.
    
    The function selects random combinations of browser and OS
    to generate diverse and legitimate-looking headers.
    """
    browsers = ["chrome", "firefox", "opera", "safari", "edge"]
    os_choices = ["win", "mac", "linux"]

    headers = Headers(
        browser=random.choice(browsers),
        os=random.choice(os_choices),
        headers=True)
    return headers.generate()

#we create an empty list that we will append with data from each URL
sales_data = []

#we create a loop to go trough each url and get the info
for sale in sales_url :
    try : 
        req = requests.get(sale, headers=get_headers())
        soup = BeautifulSoup(req.content, 'html.parser')

        """ we request the url of sale and we put the magial headers to pass the website without being blocked
        then we parse the content of the page with BeautifulSoup"""

        data_bloc = soup.find('div', class_="general-info w-100")
        if req.status_code != 200 :
            print(req.status_code)
        if not data_bloc :
            raise Exception('Data bloc not found')
        
        """We extract the part where the main infos are, in the "general info w-100" part of the html code. We ask to raise
        if the status code is not ok  (being 403 if website blocked us for example), or if the scraper was not able to find 
        the general info bloc"""

        infos = {}
        infos['url'] = sale
        elements = data_bloc.find_all(['h3', "p"])
        for i in range(0, len(elements), 2) :
            if elements[i].name == "h3" and elements[i+1].name == "p":
                key = elements[i].get_text(strip=True)
                value = elements[i+1].get_text(strip=True)
                infos[key] = value

        sales_data.append(infos)
        time.sleep(random.uniform(2, 5))

        """now we create a dictionnary infos and loop to get all the infos together, they are under the html balise <h3> for the key and 
        <p> for the value. 
        Once the loop finished, we append the sales_data list with infos and add some random time before getting the next request"""

    except Exception as e:
        print(f"Erreur avec {sale} : {e}")
    print(sales_data)

df = pd.DataFrame(data=sales_data, columns=['url', 'Données de base', 'Équipement', 'Certificats et attestations de conformité', 'Urbanisme et risques environnementaux', 'Aménagement intérieur', 'Cuisine et sanitaires', 'Description extérieure', 'Chauffage et énergie'])
print(df.columns)
df.to_csv('immovlan_raw_data_new_v2.csv', index=False, header=True, encoding='utf-8')

"""we use pandas to get the data in a csv file and get the columns. After this, we can open the file with the csv viewer to go trough the dataframe"""