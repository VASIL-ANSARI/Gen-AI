import os, csv, requests
from pypdf import PdfReader
from langchain_core.documents import Document

def load_csv(filepath: str, text_col="prompt"):
    docs = []
    if not os.path.exists(filepath):
        return docs
    with open(filepath, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_col) or ""
            docs.append(Document(page_content=text, metadata={"source": filepath}))
    return docs

def load_pdfs(pdf_folder_path: str):
    docs = []
    if not os.path.exists(pdf_folder_path):
        return docs
    for file in os.listdir(pdf_folder_path):
        if file.lower().endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(pdf_folder_path, file))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                docs.append(Document(page_content=text, metadata={"source": file}))
            except Exception:
                continue
    return docs

def load_api(api_url: str, field="content"):
    docs = []
    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                # if top-level dict contains items in a key
                if isinstance(next(iter(data.values())), list):
                    for v in next(iter(data.values())):
                        if field in v:
                            docs.append(Document(page_content=v[field], metadata={"source": api_url}))
                else:
                    # generic: convert whole json to text
                    docs.append(Document(page_content=str(data), metadata={"source": api_url}))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and field in item:
                        docs.append(Document(page_content=item[field], metadata={"source": api_url}))
                    else:
                        docs.append(Document(page_content=str(item), metadata={"source": api_url}))
    except Exception:
        pass
    return docs

def load_user_feedback(feedback_file="feedback.txt"):
    docs = []
    if not os.path.exists(feedback_file):
        return docs
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            print(f'File Text: {line}\n\n')
            if text:
                docs.append(Document(page_content=text, metadata={"source": "user_feedback"}))
    return docs
