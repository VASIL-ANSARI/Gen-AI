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

load_dotenv()  # take environment variables from .env (especially openai api key)


# Load API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()

genai.configure(api_key=api_key)

# llm = genai.GenerativeModel('gemini-2.5-flash')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


"Embeddings setup"

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)

vectordb_file_path = "vector_db_store"

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

    print("Vector database created and saved at:", vectordb_file_path)


def build_rag_chain(llm, retriever):

    template = """
    Given the following context and question, answer ONLY from the context.
    If answer is missing, say: "I don't know."

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # RAG pipeline — modern LangChain structure
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


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    rag = build_rag_chain(llm, retriever)


    return rag