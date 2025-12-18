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

## 🛠 Update API key and Dataset file path
```
In .env file, update the GOOGLE_API_KEY and dataset_file location.
```

## ▶️ How to Run
```bash
cd task_1
streamlit run main_1.py
_
```

## 🧪 Example Queries

* "What are the courses offered by Elevance Skills"
* "What is Elevance Skills"
* "Provide me contact details of Elevance Skills"
* "How to install Power BI on mac"

## Recorded Video
```
https://drive.google.com/file/d/19t7tuuEgPO7AOCPMDSFYrLVfk1HuqBpJ/view?usp=sharing
```