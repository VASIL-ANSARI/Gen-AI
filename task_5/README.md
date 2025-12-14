## 🏥 Medical Q&A Chatbot (MedQuAD)

A retrieval-augmented medical question-answering chatbot built using the MedQuAD dataset. The chatbot provides context-grounded medical information, detects user sentiment and medical anxiety, and responds using emotionally appropriate and safe language.

⚠️ Disclaimer: This chatbot is for educational purposes only and does not provide medical advice, diagnosis, or treatment.

## ✨ Key Features

📚 RAG-based Medical Q&A (FAISS + MedQuAD)

🧠 Sentiment Analysis (Positive / Neutral / Negative)

😟 Medical Anxiety Detection

📊 Sentiment Visualization with confidence score

🛡️ Strict medical safety guardrails

## 🏗️ Tech Stack

LLM: Google Gemini (via LangChain)

Embeddings: Instructor-Large (HuggingFace)

Vector Store: FAISS

NLP: Transformers + Rule-based Anxiety Detection

UI: Streamlit

Dataset: MedQuAD

## ▶️ How to Run
git clone <repo-url>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd task_5
streamlit run main_5.py

## 📥 Knowledge Base Creation

Click “Create Knowledgebase” in the UI to load the MedQuAD dataset, generate embeddings, and store vectors locally using FAISS.

## 🧠 Emotion-Aware Behavior

Negative / anxious users: Calm, empathetic responses

Positive users: Encouraging and supportive tone

Neutral users: Clear, factual, professional answers


## 🧪 Sample Questions by Sentiment
😟 Negative / Anxious

“I’m really scared about this chest pain. Is it serious?”

“I’m worried my symptoms might be cancer.”

😐 Neutral

“What are the symptoms of diabetes?”

“How is asthma diagnosed?”

“What causes high blood pressure?”

🙂 Positive

“Thanks! Can you explain how vaccines work?”