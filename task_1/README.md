# 🚀 Web Crawler & Ingestion Pipeline – Task 1  
A production-ready Python-based system that crawls any website, extracts useful content, converts it into clean Markdown files, and prepares the dataset for RAG or downstream ingestion.

## 📌 Features Implemented (Task 1 Completed)

### ✅ Website Crawling
- Crawls internal pages within the same domain  
- Extracts links and prevents infinite loops  
- Configurable depth & rate limiting

### ✅ Sitemap Parsing
- Auto-detects and parses sitemap.xml  
- Falls back to crawling if sitemap fails

### ✅ HTML → Markdown Conversion
- Removes scripts, styles, navbars, footers  
- Converts cleaned HTML into structured Markdown  
- Saves files using safe filenames

### ✅ Logging
- All events stored in crawler.log  
- Includes crawl attempts, failures, saved pages

### ✅ Configurable Pipeline
Settings stored in config.yaml

## 🛠 Installation
```
pip install -r requirements.txt
```

## 🛠 Update API key and Dataset file path
```
In config.ini, update the dataset_file location.
In .env file, update the GOOGLE_API_KEY.
```

## ▶️ How to Run
```
streamlit run main_1.py
_
```

## 📁 Output
Markdown files saved in:
```
scraped_pages/
```

## 🎯 Final Outcome
A fully working ingestion pipeline ready for RAG systems.
All the website that are crawled are stored in crawler/crawler.log file.
All the web scraping details are stored in scraped_pages folder.
vector_db_store folder contains the embeddings of the knowledge database.
The scheduler runs every 12 hours to updated the knowledge database.
