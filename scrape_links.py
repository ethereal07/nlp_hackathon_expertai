import os
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib.request 
import requests
import warnings
from fetch_article_details import get_esg_article_links

warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)

def check_path():
    return os.path.exists('./data/scraped_data.csv')

def read_data(date):
    df = get_esg_article_links(date)
    print("Extracting URLs of articles...")
    if check_path() == True:
        csv_df = pd.read_csv('./data/scraped_data.csv')
        merged_df = df.merge(csv_df, how='outer', left_on='DocumentIdentifier', right_on='url', indicator=True)
        url_df = merged_df[~((merged_df._merge == 'both') | (merged_df._merge == 'right_only'))].filter(
            ['DocumentIdentifier'], axis=1)
    else:
        url_df = df.filter(['DocumentIdentifier'], axis=1)
    print("URLs extracted!")
    return url_df

def scrape_data(date):
    url_df = read_data(date)
    df1 = pd.DataFrame(columns=['url', 'text'])
    print("Scraping text from URLs...")
    for i in url_df['DocumentIdentifier']:
        try:
            print(f"Scraping text from {i}...")
            url = i
            html = urllib.request.urlopen(url)
            htmlParse = BeautifulSoup(html, 'html.parser')
            text=''
            for para in htmlParse.find_all("p"):
                # print(para.get_text())
                text+=para.get_text()
            df1 = df1.append({'url': str(i), 'text':text},ignore_index=True)
        except:
            # print("Data couldn't be read")
            df1 = df1.append({'url': str(i), 'text':"Data couldn't be read"}, ignore_index=True)
            continue
    if check_path() == True:
        csv_df = pd.read_csv('./data/scraped_data.csv')
        final_df = pd.concat([csv_df, df1])
    else:
        final_df = df1
    final_df.to_csv('./data/scraped_data.csv' , index=False)
    print("Texts scraped!")
