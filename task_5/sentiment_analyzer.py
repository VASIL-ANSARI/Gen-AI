from transformers import pipeline
import torch

class MedicalSentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True
        )

        # Thresholds tuned for medical text
        self.NEG_THRESHOLD = 0.55
        self.POS_THRESHOLD = 0.55

    def analyze(self, text: str) -> dict:
        scores = self.sentiment_pipeline(text)[0]

        score_map = {s["label"].upper(): s["score"] for s in scores}

        positive = score_map.get("POSITIVE", 0)
        negative = score_map.get("NEGATIVE", 0)
        neutral = score_map.get("NEUTRAL", 0)

        # Decision logic (IMPORTANT)
        if negative > self.NEG_THRESHOLD:
            label = "NEGATIVE"
        elif positive > self.POS_THRESHOLD:
            label = "POSITIVE"
        else:
            label = "NEUTRAL"

        return {
            "label": label,
            "scores": {
                "positive": round(positive, 3),
                "neutral": round(neutral, 3),
                "negative": round(negative, 3)
            }
        }
