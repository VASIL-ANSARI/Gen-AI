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
* Gemini processes image + prompt together
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

## 🔐 Environment Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cd task_2
```

```env
GOOGLE_API_KEY=your_gemini_api_key_here
dataset_file=your dataset file lcoation
```

---

## ▶️ How to Run the Application

```bash
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
