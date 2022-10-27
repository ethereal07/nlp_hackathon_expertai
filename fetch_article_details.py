import gdelt
import pandas as pd
# import openpyxl
from datetime import datetime

gd = gdelt.gdelt(version=2)
pd.set_option('display.max_columns', None)

esg_list = ['EMISSIONS', 'POLLUTION', 'CARBON', 'WIND', 'SOLAR']

def get_esg_article_links(date):
    """
    This function generates all the articles having ESG themes in the given target date
    """
    formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y %b %d")
    print(f"Fetching news articles related to ESG topics for {formatted_date}...")
    results = gd.Search(formatted_date, table='gkg', coverage=False, translation=False)
    append_list = []
    for item in esg_list:
        df = results[results['V2Themes'].str.contains(item, na=False)]
        append_list.append(df)
    filtered = pd.concat(append_list)
    filtered = filtered[filtered['DocumentIdentifier'].str.startswith('https', na=False)]
    filtered = filtered[filtered['SourceCommonName'].str.endswith('com', na=False)]
    print(f"News articles fetched for {formatted_date}!")
    return filtered
