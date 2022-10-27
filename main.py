from scrape_links import scrape_data

date = '2021-11-27'

def get_text_for_nlp(art_date):
    try:
        scrape_data(art_date)
    except ValueError as e:
        print(f"No data available for {art_date}: ", e)
        pass
    
get_text_for_nlp(date)