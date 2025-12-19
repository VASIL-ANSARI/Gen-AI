from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader

from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import os
import json
from image_generation import generate

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

# LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    max_output_tokens=512
)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

vectordb_file_path = "vector_db_store"

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


prompt_template = """
You are an assistant for ElevanceSkills.

Given the following context and question, answer ONLY from the context. If answer is missing, say: "I don't know."

RULES:
- NEVER invent information.
- NEVER answer using outside knowledge.
- If a visual explanation helps, generate an IMAGE_PROMPT.
- Respond in the following JSON format:
{{
  "answer": "...",
  "image_prompt": "..."
}}

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
    return "\n\n".join(texts) 

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

    # print(response)

    raw_text = response.candidates[0].content.parts[0].text

    print(raw_text)

    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```", 2)[1]
        clean = clean.lstrip("json\n").strip()

    result = json.loads(clean)

    # result = eval(response.text)

    print(f"LLM Response: {result}\n\n")
    image_prompt = result["image_prompt"]

    answer_text = result["answer"]

    try:
        ## image generation logic
        if(image_prompt.strip() != ""):
            response_image = generate(image_prompt)
        else:
            response_image = None
    except Exception as e:
        print(e)
        response_image = None

    chatbot_response = {
        "text": answer_text,
        "image": response_image
    }
    return chatbot_response