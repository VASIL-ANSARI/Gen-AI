import os, time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
import io

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env")
    st.stop()

genai.configure(api_key=API_KEY)

st.set_page_config(page_title="ElevanceSkills Multi Modal Chatbot")

# LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    max_output_tokens=512
)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

vectordb_file_path = "vector_db_store"


## Function to load OpenAI model and get respones
def get_gemini_response(question: str, image=None):
    """
    Gemini response that considers:
    - RAG context
    - Prompt rules
    - Optional image
    """

    # 1. Retrieve context from vector DB
    context = combined_retrieval(question)

    # 2. Build final prompt using your template
    final_prompt = PROMPT.format(
        context=context,
        question=question
    )

    # 3. Gemini native model
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # 4. Multimodal call
    if image:
        response = model.generate_content(
            [final_prompt, image]
        )
    else:
        response = model.generate_content(final_prompt)

    return response.text

    # # 3. Build message correctly
    # if image:
    #     image_bytes = io.BytesIO()
    #     image.save(image_bytes, format=image.format or "PNG")
    #     image_bytes = image_bytes.getvalue()

    #     message = HumanMessage(
    #         content=[
    #             {"type": "text", "text": final_prompt},
    #             {"type": "image", "data": image_bytes},
    #         ]
    #     )
    # else:
    #     message = HumanMessage(content=final_prompt)

    # messages = [
    #     SystemMessage(content="You are an assistant for ElevanceSkills. Follow all rules strictly."),
    #     message
    # ]

    # # 4. Invoke Gemini
    # response = llm.invoke(messages)

    # return response.content



# Initialize and create vector DB
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
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

    print("Vector database created and saved at:", vectordb_file_path)

# Manual ingestion button
if st.button("🔄 Run ingestion now"):
    create_vector_db()
    st.success("Knowledge Database Updated")

st.title("ElevanceSkills — Multi Model Chatbot")

def combined_retrieval(query):
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    # Primary retrieval from main vectordb
    try:
        retriever = vectordb.as_retriever(score_threshold=0.7)
        vector_docs = retriever.invoke(query)
    except Exception:
        vector_docs = []

    # Return as text context: join top N documents' content
    texts = [d.page_content for d in vector_docs]
    return "\n\n".join(texts[:100])  # limit to reasonable size
    
prompt_template = """
You are an assistant for ElevanceSkills.

Given the following context and question, answer ONLY from the context. If answer is missing, say: "I don't know."

RULES:
- NEVER invent information.
- NEVER answer using outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}
"""

PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

rag = (
    RunnableMap({
        "question": RunnablePassthrough(),
        "context": RunnableLambda(lambda q: combined_retrieval(q))
    })
    | PROMPT
    | llm
    | StrOutputParser()
)

# UI: query box
input = st.text_input("Ask Question: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
# query = st.text_input("Ask about ElevanceSkills:")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


if st.button("Ask"):
    with st.spinner("Answering..."):
        response = get_gemini_response(input, image)
        # answer = rag.invoke(query)


    st.subheader("Answer")
    st.write(response)

# Show answer & feedback if available
# if st.session_state.get("show_feedback", False):
#     st.subheader("Answer")
#     st.write(st.session_state["last_answer"])

#     # Store feedback init entry and provide feedback id
#     st.write("---")
#     st.markdown("### Feedback (help improve the bot):")
#     rating = st.radio("Was this answer helpful?", ("Yes", "No"), key="rating")
#     correction = st.text_area("Correction / additional info", key="correction")

#     if st.button("Submit feedback") and correction:
#         with open("feedback.txt", "a", encoding="utf-8") as f:
#             f.write(
#                 f"QUESTION: {st.session_state['last_query']}\n"
#                 f"ANSWER: {st.session_state['last_answer']}\n"
#                 f"RATING: {st.session_state['rating']}\n"
#                 f"CORRECTION: {st.session_state['correction']}\n---\n"
#             )
#         st.success("Thanks — feedback recorded.")
