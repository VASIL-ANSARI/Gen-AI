ANXIETY_KEYWORDS = [
    "worried", "scared", "afraid", "anxious", "panic",
    "serious", "emergency", "life threatening",
    "pain", "bleeding", "cancer", "tumor", "heart attack"
]

def detect_medical_anxiety(text: str, sentiment_data: dict) -> bool:
    text_lower = text.lower()

    keyword_match = any(word in text_lower for word in ANXIETY_KEYWORDS)
    negative_sentiment = sentiment_data["label"].lower() == "negative"
    high_confidence = sentiment_data["scores"][sentiment_data["label"].lower()] >= 0.65

    # Strong signal for medical anxiety
    return keyword_match and negative_sentiment and high_confidence
