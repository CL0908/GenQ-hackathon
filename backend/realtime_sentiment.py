import requests
from transformers import pipeline

url = 'https://newsapi.org/v2/everything'

params = {
    'q' : '"sustainability policy" OR "carbon trading" OR "green regulation"',
    'language': 'en',
    'apiKey': 'df1369c2201b4d1db44123106e76833d',
    'sortBy' : 'relevancy'
}

response = requests.get(url, params = params)
data = response.json()
if 'articles' in data:
    articles = data['articles']
else:
    print("⚠️  'articles' key not found in API response.")
    print("Status code:", response.status_code)
    print("Full response:")
    import json
    print(json.dumps(data, indent=2))
    articles = []



##Classifier
topic_classifier = pipeline("zero-shot-classification", model = 'facebook/bart-large-mnli')
sentiment_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

candidate_labels = ["sustainability policy", "carbon trading", "environmental", "greentech",  "unrelated" ]

##Analyze article
results = []

for article in articles:
        text = article["title"] + "." + (article.get("description") or "")  ## get the article detatils for later sentiment analysis

        
        topic = topic_classifier(text, candidate_labels)
        top_label = topic["labels"][0] ## get the highest confidence score
        if top_label in ["sustainability policy", "carbon trading", "environmental", "greentech"]:
                

                ##sentiment analysis of the filtered news

                sentiment = sentiment_classifier(text)[0]
                label = sentiment["label"]
                score = sentiment["score"]

                 
                if label == "positive":
                    impact = score * 1       #positive impact
                elif label == "negative":
                    impact = score * -1      #negative impact
                else:
                    impact = 0               #neutral impact

                results.append({
                        "title": article["title"],
                        "url": article["url"],
                        "sentiment": label,
                        "impact_score": round(impact, 3)
                    })
                
##Display the result
for i, r in enumerate(results, 1):
       print(f"{i}. {r['title']}")
       print(f"   Sentiment: {r['sentiment']}, Impact Score: {r['impact_score']}")
       print(f"   URL: {r['url']}\n")

# Step 5: Aggregate market signal
if results:
    market_index = sum(r["impact_score"] for r in results) / len(results)
    print(f" Overall GreenTech Market Sentiment Index: {round(market_index, 3)}")
    print("--lemonwee")
else:
    print("No relevant sustainability articles found.")
                


