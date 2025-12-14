# 🩺 Medical Q&A Chatbot using MedQuAD

A specialized **Medical Question-Answering Chatbot** built using the **MedQuAD dataset**. The application supports medical information retrieval through semantic search, basic medical entity recognition, and an easy-to-use **Streamlit** interface.

> ⚠️ **Disclaimer**: This chatbot is for **educational and informational purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**.

---

## 🚀 Features

* ✅ Medical question-answering using **MedQuAD dataset**
* ✅ Semantic search with **FAISS vector database**
* ✅ Retrieval-Augmented Generation (RAG)
* ✅ Basic medical entity awareness (symptoms, diseases, treatments)
* ✅ Streamlit-based user interface
* ✅ Fast and lightweight inference
* ✅ Extensible for advanced medical NLP tasks

---

## 🧠 Dataset

**MedQuAD (Medical Question Answering Dataset)**

* Source: [https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)
* Curated medical Q&A pairs from trusted sources (NIH, CancerGov, etc.)
* XML format converted into CSV for easier processing

### CSV Format Used

```csv
prompt,response
"What are the symptoms of leukemia?","Symptoms include fatigue, fever, weight loss..."
```

* `prompt` → medical question
* `response` → medical answer

---

## 🏗️ Architecture Overview

```
User → Streamlit UI → Retriever (FAISS)
                      ↓
               Relevant Medical Context
                      ↓
                  LLM Response
```

---

## 📁 Project Structure

```
medical-qa-chatbot/
│
├── data/
│   ├── medquad.csv          # Processed dataset
│
├── ingest.py                # XML → CSV conversion
├── main_3.py                   # Streamlit application
└── README.md                # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/medical-qa-chatbot.git
cd medical-qa-chatbot
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\\Scripts\\activate    # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
cd task_3
```

---

## 📥 Dataset Preparation

### Download MedQuAD

```bash
git clone https://github.com/abachaa/MedQuAD.git
```

Place the `MedQuAD` folder in the project root.

---

## 🗂️ Data Ingestion

Convert MedQuAD XML files into CSV:

```bash
python ingest.py
```

This generates:

```
data/medquad.csv
```
 Note: The csv file is already generated. Follow below steps to run the application.


---

## ▶️ Run the Application

```bash
streamlit run main_3.py
```

Click on Create Knowledgebase
This creates FAISS embeddings stored locally.


---

## 🧪 Example Questions

* "What are the symptoms of breast cancer?"
* "How is diabetes treated?"
* "What causes leukemia?"
* "What tests diagnose lung cancer?"

---