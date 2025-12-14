import streamlit as st
from helper import create_vector_store, retrieve_context
from sentiment import detect_sentiment

st.set_page_config(page_title="Sentiment-Medical Q&A Chatbot")

st.title("🏥 Sentiment - Medical Q&A Chatbot (MedQuAD)")
st.warning("⚠️ This chatbot is for educational purposes only.")

btn = st.button("Create Knowledgebase")
if btn:
    print("Loading data...")
    create_vector_store()
    print("Loaded Data Successfully")
    st.success("Knowledge Database Created")

question = st.text_input("Ask a question:")

if st.button("Ask") and question:
    with st.spinner("Analyzing your question..."):

        # 1. Sentiment detection
        sentiment_data = detect_sentiment(question)

        # 2. RAG response
        rag = retrieve_context(sentiment_data['sentiment'])

        response = rag.invoke(question)

    st.subheader("📌 Answer")
    st.write(response)

    st.subheader("💬 Sentiment Detected")
    st.json(sentiment_data)