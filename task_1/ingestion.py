from hash_utils import load_hashes, save_hashes, compute_hash
from crawler import discover_site
from scrapers import load_csv, load_pdfs, load_api, load_user_feedback
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Access values
csv_path = config["PATHS"]["dataset_file"]

def deduplicate_documents(docs, ingested_hashes):
    new_docs = []
    new_hashes = set()
    for d in docs:
        h = compute_hash(d.page_content)
        if h not in ingested_hashes:
            d.metadata["hash"] = h
            new_docs.append(d)
            new_hashes.add(h)
    return new_docs, new_hashes

def ingest_all_sources(vectordb_path, embeddings,
                       site_urls=None, pdf_folder="company_docs",
                       csv_path=csv_path, api_urls=None, feedback_file="feedback.txt"):
    """
    - vectordb_path: folder path for local FAISS
    - embeddings: an embeddings object (HuggingFaceInstructEmbeddings)
    - site_urls: list of seed site urls to crawl (e.g. ["https://www.elevanceskills.com"])
    - pdf_folder, csv_path, api_urls: other sources
    """
    if site_urls is None:
        site_urls = ["https://www.elevanceskills.com/"]

    ingested_hashes = load_hashes()

    all_docs = []

    # 1) Crawl & scrape site(s)
    for base in site_urls:
        try:
            res = discover_site(base, max_pages=150, delay=0.8)
            manifest = res.get("manifest", {})
            # convert saved files to Documents
            for url, meta in manifest.items():
                fpath = meta.get("file")
                if fpath:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    all_docs.append(Document(page_content=content, metadata={"source": url}))
        except Exception:
            pass

    # 2) CSV FAQ
    try:
        csv_docs = load_csv(csv_path)
        all_docs.extend(csv_docs)
    except Exception:
        pass

    # 3) PDFs local folder
    try:
        pdf_docs = load_pdfs(pdf_folder)
        all_docs.extend(pdf_docs)
    except Exception:
        pass

    # 4) API endpoints
    if api_urls:
        for u in api_urls:
            try:
                api_docs = load_api(u)
                all_docs.extend(api_docs)
            except Exception:
                pass

    # 5) User feedback
    try:
        fb_docs = load_user_feedback(feedback_file)
        all_docs.extend(fb_docs)
    except Exception:
        pass

    # Deduplicate
    new_docs, new_hashes = deduplicate_documents(all_docs, ingested_hashes)

    if not new_docs:
        print("No new docs to ingest.")
        return

    # Load existing vectordb or create new
    if os.path.exists(vectordb_path):
        vectordb = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(new_docs)
        vectordb.save_local(vectordb_path)
        print(f"Appended {len(new_docs)} docs to existing vector DB.")
    else:
        vectordb = FAISS.from_documents(documents=new_docs, embedding=embeddings)
        vectordb.save_local(vectordb_path)
        print(f"Created new vector DB with {len(new_docs)} docs.")

    ingested_hashes.update(new_hashes)
    save_hashes(ingested_hashes)
    print("Ingestion complete.")
