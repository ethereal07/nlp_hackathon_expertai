## Steps in this Project

### Usage of GDELT data to fetch relevant news articles for ESG topics

- We use gdelt python package to select articles which cover ESG related news
- Gdelt version 2 has been used in this project
- The articles are being fetched for a particular date (which is an input from the user)
- We are fetching only the URLs from the returned dataframe. Only english news articles are being selected
- Articles are selected based on keywords and phrases in a given time frame

### Usage of BeautifulSoup to scrape text from the news articles
- BeautifulSoup python package is being used to scrape texts from the news articles' URLs
- The scraped text from the article for the particular user-specified date is then stored as a dataframe
- The text is then appended and stored in a csv file upon which expert.ai's Natural Language API is being applied


