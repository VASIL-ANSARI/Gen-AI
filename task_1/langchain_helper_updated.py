import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import json

from crawler import discover_site

load_dotenv()  # take environment variables from .env (especially openai api key)

# setting up logger...

import logging

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add formatter to handler
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Load API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()

genai.configure(api_key=api_key)

# llm = genai.GenerativeModel('gemini-2.5-flash')

llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    temperature=0.2,
)


"Embeddings setup"

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)

vectordb_file_path = "/Users/vasilansari/Desktop/Gen AI Project/customer-chatbot/task_1/vector_db_store"

def create_vector_db():
    dataset_path = "/Users/vasilansari/Desktop/Gen AI Project/customer-chatbot/dataset/dataset.csv" 
    
    # Load data from FAQ sheet
    # Load CSV
    loader = CSVLoader(
        file_path=dataset_path,
        source_column="prompt",   # column to extract data from
        encoding="latin-1"
    )

    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

    fetch_and_ingest_site()

    print("Vector database created and saved at:", vectordb_file_path)

def fetch_and_ingest_site():
    base = "https://www.elevanceskills.com"
    result = discover_site(base_url=base, max_pages=200, delay=0.8)

    # result contains:
    #   - manifest: mapping url -> saved text file + metadata
    #   - pdfs: list of downloaded pdf paths
    #   - endpoints: list of detected XHR/API endpoints

    # 1) Convert saved page text files into LangChain documents
    docs = []
    for url, meta in result["manifest"].items():
        fpath = meta.get("file")
        if not fpath or not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            txt = f.read()
        # create a Document-like object for FAISS:
        docs.append(Document(page_content=txt, metadata={"source": url}))

    # 2) Add PDF files to docs (LangChain PyPDFLoader can be used too)
    for pdf_path in result["pdfs"]:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            docs.extend(pages)
        except Exception as e:
            logger.warning(f"PDF parse failed {pdf_path}: {e}")

    # 3) Add to FAISS (append)
    if docs:
        print(f'Docs : {docs}\n\n')
        try:
            vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
            vectordb.add_documents(docs)
            vectordb.save_local(vectordb_file_path)
        except Exception:
            vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)
            vectordb.save_local(vectordb_file_path)

    # 4) Save discovered API endpoints somewhere for manual review / automated calls
    with open("scraped_pages/discovered_endpoints.json", "w", encoding="utf-8") as f:
        json.dump(result["endpoints"], f, indent=2)

    return result

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=1.0)

    # get information from other sources
    # def combined_retrieval(question):
    #     all_docs = []

    #     docs = retriever.invoke(question)
    #     # print(f'{docs}\n\n')

    #     all_docs.extend(d.page_content for d in docs)
    #     # print(f'{all_docs}\n\n')
        
    #     fallback_docs = get_fallback_context()
    #     # print(f'{fallback_docs}\n\n')

    #     all_docs.extend(fallback_docs)

    #     # docs = retriever.invoke(question)

    #     # print(f'Docs : {all_docs}\n\n')

    #     return all_docs

        # If vector DB fails → fallback
        # if not docs or len(docs) == 0:
        #     print("Primary DB retrieval failed. Using fallback sources...")
        #     return get_fallback_context()

        # Vector DB returned something
        # return "\n\n".join(d.page_content for d in docs)
    

    template = """
        You are an ElevanceSkills AI assistant.

        Retrieval Strategy:
        - First, try answering ONLY using the CONTEXT.
        - If context was pulled from fallback sources (website, PDFs, APIs, user feedback), mention it.
        - If the answer is still unavailable, reply:
        "I don't know based on current data."

        CONTEXT:
        {context}

        QUESTION:
        {question}
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    rag_chain = (
        RunnableMap({
            "question": RunnablePassthrough(),
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain