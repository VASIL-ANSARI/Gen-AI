import os, time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from scheduler import start_scheduler
from ingestion import ingest_all_sources
from scrapers import load_user_feedback

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env")
    st.stop()

genai.configure(api_key=API_KEY)

st.set_page_config(page_title="ElevanceSkills RAG Chatbot")

# LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

VECTORDB_PATH = "vector_db_store"

#create feedback file to store user feedback
if not os.path.exists("feedback.txt"):
    with open("feedback.txt", "w") as f:
        f.write("")  # empty file
    print("File created.")
else:
    print("Feeback File already exists.")

# Initialize / load vector DB
if os.path.exists(VECTORDB_PATH):
    vectordb = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    # create a tiny placeholder DB to avoid errors - can be replaced by first ingestion
    placeholder = [Document(page_content="placeholder", metadata={"source": "init"})]
    vectordb = FAISS.from_documents(documents=placeholder, embedding=embeddings)
    vectordb.save_local(VECTORDB_PATH)

# Manual ingestion button
if st.button("🔄 Run ingestion now"):
    # Start scheduler (background)
    start_scheduler(VECTORDB_PATH, embeddings, site_urls=["https://www.elevanceskills.com/"])
    # ingest_all_sources(VECTORDB_PATH, embeddings, site_urls=["https://www.elevanceskills.com/"])
    st.success("Knowledge Database ingestion started")

st.title("ElevanceSkills — Dynamic RAG Chatbot")

def get_feedback_docs(query, top_k=10):
    # simple feedback retrieval: load feedback file and return all entries
    # optional: build a small feedback vectorstore separately to do similarity search
    feedback_docs = load_user_feedback("feedback.txt")
    # return the most recent/top_k (simple)
    return feedback_docs[:top_k]

def combined_retrieval(query):
    # Primary retrieval from main vectordb
    try:
        retriever = vectordb.as_retriever(score_threshold=0.8)
        vector_docs = retriever.invoke(query)
    except Exception:
        vector_docs = []

    # Always include feedback docs
    feedback_docs = get_feedback_docs(query)

    # If vector_docs empty, fallback to quick live scrape (lightweight) using discover site for the specific page is heavy.
    if not vector_docs:
        # fallback: include feedback only and a short note (we avoid re-crawling heavy)
        combined = feedback_docs
    else:
        combined = vector_docs + feedback_docs

    # Return as text context: join top N documents' content
    texts = [d.page_content for d in combined]
    return "\n\n".join(texts[:12])  # limit to reasonable size


def feedback_wrapper(query):
    # ALWAYS return feedback docs (text merged)
    docs = get_feedback_docs(query)
    if not docs:
        return "No relevant user feedback."
    return "\n\n".join([d.page_content for d in docs])
    
prompt_template = """
You are an assistant for ElevanceSkills.

You have multiple knowledge sources. Use them in this strict priority order:

1. PRIMARY CONTEXT (highest priority)
   - These are verified source of information.
   - Company documents, scraped website pages, PDFs, APIs, and other ingested data.

2. USER FEEDBACK 
   - These are corrections provided by users.
   - If any feedback matches or relates to the question, try to verify it from primary context and then provide answer.

3. If the information is not present in either user feedback or primary context, say:
   "I don't know based on current data."

RULES:
- NEVER invent information.
- NEVER answer using outside knowledge.

USER FEEDBACK CONTEXT:
{feedback_context}

PRIMARY CONTEXT:
{context}

QUESTION:
{question}
"""

PROMPT = PromptTemplate(input_variables=["context", "feedback_context", "question"], template=prompt_template)

rag = (
    RunnableMap({
        "question": RunnablePassthrough(),
        "context": RunnableLambda(lambda q: combined_retrieval(q)),
        "feedback_context": RunnableLambda(lambda q: feedback_wrapper(q))
    })
    | PROMPT
    | llm
    | StrOutputParser()
)

# UI: query box
query = st.text_input("Ask about ElevanceSkills:")

if st.button("Ask") and query:
    with st.spinner("Answering..."):
        answer = rag.invoke(query)

    # Store state
    st.session_state["last_query"] = query
    st.session_state["last_answer"] = answer
    st.session_state["show_feedback"] = True

    # st.subheader("Answer")
    # st.write(answer)

# Show answer & feedback if available
if st.session_state.get("show_feedback", False):
    st.subheader("Answer")
    st.write(st.session_state["last_answer"])

    # Store feedback init entry and provide feedback id
    st.write("---")
    st.markdown("### Feedback (help improve the bot):")
    rating = st.radio("Was this answer helpful?", ("Yes", "No"), key="rating")
    correction = st.text_area("Correction / additional info", key="correction")

    if st.button("Submit feedback") and correction:
        with open("feedback.txt", "a", encoding="utf-8") as f:
            f.write(
                f"QUESTION: {st.session_state['last_query']}\n"
                f"ANSWER: {st.session_state['last_answer']}\n"
                f"RATING: {st.session_state['rating']}\n"
                f"CORRECTION: {st.session_state['correction']}\n---\n"
            )
        st.success("Thanks — feedback recorded.")
