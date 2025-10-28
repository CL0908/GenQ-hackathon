import json
import pandas as pd
import requests
from transformers import pipeline
from datetime import datetime


with open("marketprice.json", "r") as f:
    stock_data = json.load(f)

stock_df = pd.DataFrame(list(stock_data.items()), columns=["Date", "Close"])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df = stock_df.sort_values("Date").reset_index(drop=True)

#Fetch news for each date

api_key = ""
query = '"sustainability policy" OR "carbon trading" OR "green technology"'

def fetch_news_for_date(api_key, query, date_str):
    """
    Fetch news for a specific date (returns list of titles + descriptions)
    """
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': date_str,
        'to': date_str,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 20,
        'apiKey': 'ec2d06261fb44f6ba9febd61899000a6'
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Warning: API request failed for {date_str}")
        return []
    articles = response.json().get('articles', [])
    return [a['title'] + ". " + (a.get('description') or "") for a in articles]

# Fetch news for all dates
stock_df['News'] = stock_df['Date'].dt.strftime("%Y-%m-%d").apply(
    lambda d: fetch_news_for_date(api_key, query, d)
)


#Initialize sentiment model

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # faster than roberta

def calculate_daily_sentiment(news_list):
    if not news_list:
        return 0.0  # no news → neutral
    results = sentiment_model(news_list)  # batch processing
    scores = []
    for sentiment in results:
        label = sentiment['label']
        score = sentiment['score']
        if label.lower() == "positive":
            impact = score
        elif label.lower() == "negative":
            impact = -score
        else:
            impact = 0
        scores.append(impact)
    return sum(scores) / len(scores)

stock_df['Sentiment_Score'] = stock_df['News'].apply(calculate_daily_sentiment)


stock_df['Sentiment_Score'] = stock_df['Sentiment_Score'].ffill()


stock_df.to_csv("stock_with_sentiment.csv", index=False)

print("CSV saved as stock_with_sentiment.csv")

