from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def load_ner_pipeline():
    """
    Load a biomedical NER model from HuggingFace.
    This works on any Mac M1/M2/M3/M4 without compilation.
    """
    model_name = "d4data/biomedical-ner-all"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_medical_entities(text: str):
    """
    Extract medical entities from text using BioClinicalBERT-based NER.
    Returns dict with categorized entities.
    """
    if not text.strip():
        return {}

    ner_pipeline = load_ner_pipeline()
    output = ner_pipeline(text)

    entities = {
        "diseases": [],
        "symptoms": [],
        "drugs": [],
        "treatments": [],
        "anatomy": [],
        "others": []
    }

    for item in output:
        label = item["entity_group"].lower()
        ent = item["word"]

        if "disease" in label:
            entities["diseases"].append(ent)
        elif "symptom" in label:
            entities["symptoms"].append(ent)
        elif "drug" in label or "chemical" in label:
            entities["drugs"].append(ent)
        elif "treatment" in label or "procedure" in label:
            entities["treatments"].append(ent)
        elif "anatomy" in label:
            entities["anatomy"].append(ent)
        else:
            entities["others"].append(ent)

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities