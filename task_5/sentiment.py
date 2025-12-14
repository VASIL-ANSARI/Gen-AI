from transformers import pipeline

# Lightweight & accurate model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def detect_sentiment(text: str) -> dict:
    """
    Returns:
    {
        "label": POSITIVE | NEGATIVE | NEUTRAL,
        "confidence": float
    }
    """
    result = sentiment_analyzer(text[:512])[0]

    score = round(result["score"], 2)
    
    label = result["label"]

    if label == "NEGATIVE" and score >= 0.60:
        sentiment = "negative"
    elif label == "POSITIVE" and score >= 0.65:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "confidence": score
    }
