import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title(" CUSTOMER SERVICE CHATBOT 🤖")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    rag = get_qa_chain()
    response = rag.invoke(question)

    st.header("Answer")
    st.write(response)