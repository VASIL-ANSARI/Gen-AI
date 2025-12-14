import streamlit as st
from helper import create_vector_store, retrieve_context
from language_utils import detect_language
from translator import translate


st.set_page_config(page_title="Multilingual Medical Q&A Chatbot")

st.title("🏥 Multilingual Medical Q&A Chatbot (MedQuAD)")
st.warning("⚠️ This chatbot is for educational purposes only.")

btn = st.button("Create Knowledgebase")
if btn:
    print("Loading data...")
    create_vector_store()
    print("Loaded Data Successfully")
    st.success("Knowledge Database Created")

question = st.text_input("Ask a question:")

if st.button("Ask") and question:
    with st.spinner("Understanding your question..."):

        # 1. Detect language
        user_lang = detect_language(question)

        # 2. Translate to English for retrieval
        translated_question = translate(
            question,
            source=user_lang,
            target="en"
        )

        # 3. Get answer from RAG
        rag = retrieve_context()
        english_response = rag.invoke(translated_question)

        # 5. Translate response back
        final_response = translate(
            english_response,
            source="en",
            target=user_lang
        )

    st.subheader("📌 Answer")
    st.write(final_response)

    st.caption(f"Language detected: {user_lang.upper()}")
