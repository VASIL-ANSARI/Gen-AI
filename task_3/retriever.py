import json
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ner import extract_medical_entities
import os
import traceback


load_dotenv()  # take environment variables from .env (especially openai api key)

# Load API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
)

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb_file_path = "vector_db_store"

def create_vector_store():
    dataset_path = "data/medquad.csv" 
    
    try:
        # Load CSV
        loader = CSVLoader(
            file_path=dataset_path,
            source_column="prompt",   # column to extract data from
            encoding="latin-1"
        )

        data = loader.load()

        print("Step 2: Loaded", len(data), "documents")

        print("Step 3: Initializing embeddings...")

        # Create a FAISS instance for vector database from 'data'
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)

        print("Vector database created and saved at: ", vectordb_file_path)
    except Exception as e:
        print("ERROR OCCURRED:")
        print(traceback.format_exc())
        raise


def retrieve_context():
    # embeddings = SentenceTransformer(EMBED_MODEL)

    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # vectordb = FAISS.load_local(
    #     "vectorstore/faiss_index",
    #     embeddings,
    #     allow_dangerous_deserialization=True
    # )

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.8)

    rag = build_rag_chain(retriever)

    return rag

def build_rag_chain(retriever):

    template = """
    Given the following context and question, answer ONLY from the context.
    If answer is missing, say: "I don't know based on the available medical data."

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

"""
Complete wrapper with RAG pipeline & NER extraction.
"""
def answer_with_entities(question: str):
    
    rag = retrieve_context()
    response = rag.invoke(question)

    query_entities = extract_medical_entities(question)
    answer_entities = extract_medical_entities(response)

    return {
        "answer": response,
        "query_entities": query_entities,
        "answer_entities": answer_entities
    }