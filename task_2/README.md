# ElevanceSkills Multimodal RAG Chatbot

A **Streamlit-based multimodal chatbot** powered by **Google Gemini**, **LangChain**, and **FAISS**, designed to answer user questions **strictly from a custom knowledge base** (CSV) with optional **image understanding**.

The application combines:

* **Retrieval-Augmented Generation (RAG)** for factual accuracy
* **Google Gemini** for reasoning and multimodal understanding
* **FAISS vector database** for fast semantic search
* **Streamlit UI** for an interactive user experience

---

## 🚀 Key Features

### 1. Retrieval-Augmented Generation (RAG)

* Uses a CSV-based knowledge source
* Converts content into embeddings using **HuggingFace Instructor embeddings**
* Stores and retrieves data via **FAISS**
* Ensures answers are grounded in provided context only

### 2. Hallucination-Safe Responses

* Enforced prompt rules:

  * Answer **only from retrieved context**
  * If information is missing, respond with **"I don't know"**
  * No outside knowledge allowed

### 3. Multimodal Capabilities

* Accepts **text queries**
* Accepts **image uploads (JPG, PNG, JPEG)**
* Gemini processes image + prompt tog
ether
* Useful for chart explanations, visual interpretation, and UI screenshots

### 4. On-Demand Knowledge Ingestion

* Manual button to regenerate vector database
* Enables easy dataset updates without code changes

### 5. Clean & Extensible Architecture

* LangChain for orchestration
* Gemini SDK for multimodal generation
* Easily extendable with memory, citations, or feedback loops

---

## 🧠 Architecture Overview

```
CSV Dataset
   ↓
Document Loader
   ↓
Embeddings (Instructor)
   ↓
FAISS Vector DB
   ↓
Context Retrieval
   ↓
Prompt Template (Rules + Context)
   ↓
Gemini (Text / Image)
   ↓
Streamlit UI
```

---

## ▶️ How to Run the Application

```bash
cd task_2
streamlit run main_2.py
```

---

## 🔄 Creating / Updating the Vector Database

1. Launch the app
2. Click:

```
🔄 Run ingestion now
```

This will:

* Load CSV
* Generate embeddings
* Create FAISS index
* Save it locally for future queries

---

## 💬 How to Use

1. Enter your **text question** in the input box
2. (Optional) Upload an **image**
3. Click **Ask**
4. Receive an answer grounded strictly in your dataset

If the answer is not found in the context, the bot will respond:

```
I don't know.
```

---

## 📦 Key Technologies Used

| Component        | Technology             |
| ---------------- | ---------------------- |
| UI               | Streamlit              |
| LLM              | Google Gemini 2.5      |
| RAG Framework    | LangChain              |
| Vector DB        | FAISS                  |
| Embeddings       | HuggingFace Instructor |
| Image Processing | Pillow                 |
| Config           | python-dotenv          |

---


## Example Queries
* What is the difference between them. (Attach img3)
* How can I use PowerBI on my Mac system?
* What is the issue in this SQL query? (Attach img2)

## Recorded Video
```
https://drive.google.com/file/d/1BLk1hIhZRfT_WtVYGBACKe4w_UT9Bwvq/view?usp=sharing
```
