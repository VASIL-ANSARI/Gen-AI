import threading, time
from ingestion import ingest_all_sources

from dotenv import load_dotenv
import os

load_dotenv()
csv_path = os.getenv("dataset_file")

print(csv_path)


def start_scheduler(vectordb_path, embeddings, site_urls=None, pdf_folder="company_docs", csv_path=csv_path, api_urls=None, feedback_file="feedback.txt"):
    def run():
        while True:
            try:
                print("Scheduler: running ingestion...")
                ingest_all_sources(vectordb_path, embeddings, site_urls=site_urls, pdf_folder=pdf_folder, csv_path=csv_path, api_urls=api_urls, feedback_file=feedback_file)
            except Exception as e:
                print("Scheduler ingestion error:", e)
            time.sleep(12 * 60 * 60)  # 12 hours
    t = threading.Thread(target=run, daemon=True)
    t.start()
