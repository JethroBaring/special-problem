from transformers import pipeline

# Load sentiment model once (to avoid loading delay)
sentiment_model = pipeline("sentiment-analysis")

async def analyze_sentiment(text):
    try:
        result = sentiment_model(text)[0]
        sentiment = result["label"]
        confidence = result["score"]

        return {
            "sentiment": sentiment,
            "confidence": confidence
        }

    except Exception as e:
        print(f"⚠️ Sentiment analysis error: {e}")
        return None
