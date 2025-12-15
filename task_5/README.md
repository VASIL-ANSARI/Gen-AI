## 🏥 Medical Q&A Chatbot (MedQuAD)

A retrieval-augmented medical question-answering chatbot built using the MedQuAD dataset.

The chatbot combines retrieval-augmented generation with domain-specific sentiment analysis and confidence visualization to improve emotional awareness, transparency, and response appropriateness in medical conversations.

⚠️ Disclaimer: This chatbot is for educational purposes only and does not provide medical advice, diagnosis, or treatment.

## ✨ Key Features

📚 RAG-based Medical Q&A (FAISS + MedQuAD)

🧠 Sentiment Analysis (Positive / Neutral / Negative)

😟 Medical Anxiety Detection

📊 Sentiment Visualization with confidence score

🛡️ Strict medical safety guardrails

## Sentiment Analyzer Model Evaluation
We fine-tuned a ClinicalBERT model on medical-domain sentiment data to reduce false negative bias caused by clinical terminology. This significantly improved neutral sentiment detection and emotional sensitivity in healthcare interactions.

## 🏗️ Tech Stack

LLM: Google Gemini (via LangChain)

Embeddings: Instructor-Large (HuggingFace)

Vector Store: FAISS

NLP: Transformers + Rule-based Anxiety Detection

UI: Streamlit

Dataset: MedQuAD

## ▶️ How to Run

```bash
git clone <repo-url>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd task_5
streamlit run main_5.py
```

## 📥 Knowledge Base Creation

Click “Create Knowledgebase” in the UI to load the MedQuAD dataset, generate embeddings, and store vectors locally using FAISS.

## 🧠 Emotion-Aware Behavior

Negative / anxious users: Calm, empathetic responses

Positive users: Encouraging and supportive tone

Neutral users: Clear, factual, professional answers


## 🧪 Sample Questions by Sentiment
😟 Negative / Anxious

“What are the symptoms of leukemia?”

“I am really scared about my diagnosis”

😐 Neutral

“Can chemotherapy cause fatigue?”

🙂 Positive

“Thanks! Can you explain how vaccines work?”