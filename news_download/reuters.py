import newspaper
from newspaper import Article

# 1. Target a specific conflict news hub
topic_url = 'https://www.reuters.com/world/iran/'
topic_url = 'https://www.reuters.com/world/iran/'
topic_url = 'https://www.bbc.com/news/topics/cx2jyv8j8gwt'
# 2. Build the source object to find all active links on that page
news_hub = newspaper.build(topic_url, memoize_articles=False)

# 3. Download and parse the first available article as a test
if news_hub.articles:
    target_article = news_hub.articles[0]
    target_article.download()
    target_article.parse()
    
    print("Title:", target_article.title)
    print("Text snippet:", target_article.text[:300])
else:
    print("No active articles found on this landing page layout.")
