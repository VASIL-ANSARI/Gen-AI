import streamlit as st
from retriever import retrieve_context, create_vector_store

st.set_page_config(page_title="Medical Q&A Chatbot")

st.title("🏥 Medical Q&A Chatbot (MedQuAD)")
st.warning("⚠️ This chatbot is for educational purposes only.")

btn = st.button("Create Knowledgebase")
if btn:
    print("Loading data...")
    create_vector_store()
    print("Loaded Data Successfully")
    st.success("Knowledge Database Created")

question = st.text_input("Ask a question:")

if st.button("Ask") and question:
    with st.spinner("Searching medical knowledge..."):
        rag = retrieve_context()
        response = rag.invoke(question)

    st.subheader("Answer")
    st.write(response)
