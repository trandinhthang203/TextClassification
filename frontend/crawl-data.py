import requests
from bs4 import BeautifulSoup
import time

def get_article_links(topic_url, num_pages=1):
    links = []
    for page in range(1, num_pages + 1):
        url = f"{topic_url}-p{page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = soup.find_all('h3', class_='title-news')
        for article in articles:
            a_tag = article.find('a')
            if a_tag and 'href' in a_tag.attrs:
                links.append(a_tag['href'])
        
        time.sleep(1)
    return links

def get_article_content(article_url):
    response = requests.get(article_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find('h1', class_='title-detail')
    title = title.get_text(strip=True) if title else 'Không có tiêu đề'

    paragraphs = soup.find_all('p', class_='Normal')
    content = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    return title, content

def crawl_topic(topic_url, num_pages=1):
    article_links = get_article_links(topic_url, num_pages)
    articles = []
    
    for link in article_links:
        try:
            title, content = get_article_content(link)
            articles.append({'url': link, 'title': title, 'content': content})
            print(f"Crawled: {title}")
            time.sleep(1)
        except Exception as e:
            print(f"Error with {link}: {e}")
    return articles

if __name__ == "__main__":
    topic_url = "https://vnexpress.net/thoi-su"
    result = crawl_topic(topic_url, num_pages=2)

    # In kết quả
    for idx, article in enumerate(result, 1):
        print(f"\n📰 Bài {idx}: {article['title']}")
        print(article['content'][:500] + "...")
