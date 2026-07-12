from newspaper import Article

# 1. Provide the URL of the news article
url = 'https://edition.cnn.com/2023/06/10/sport/manchester-city-wins-champions-league-for-first-time-beating-inter-milan-1-0-in-tense-istanbul-final/index.html'
url = 'https://www.reuters.com/business/energy/oil-edges-lower-heads-weekly-gain-middle-east-supply-risks-persist-2026-07-10/'
url = 'https://www.cnn.com/world/middleeast/iran'
url = 'https://www.bbc.com/news/topics/cx2jyv8j8gwt'
url = 'https://professorrobertpape.substack.com/p/nato-changed-the-ukraine-war-what'
# 2. Create the Article object
article = Article(url)

# 3. Download and parse the article
article.download()
article.parse()

# 4. Extract the desired information
print("Title:", article.title)
print("Authors:", article.authors)
print("Publish Date:", article.publish_date)
print("Article Text:", article.text[:200]) # Printing first 200 characters
print("Article Text:", article.text) # Printing the entire text

