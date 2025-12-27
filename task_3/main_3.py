import streamlit as st
from retriever import answer_with_entities, create_vector_store

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
        response = answer_with_entities(question)
        # response = rag.invoke(question)

    st.subheader("Answer")
    st.write(response["answer"])

    #NER
    st.markdown("---")
    st.subheader("🔍 Detected Medical Entities")

    q_e = response["query_entities"]
    a_e = response["answer_entities"]

    st.write("[Entities] From your question:")
    st.write(f"**Diseases:** {', '.join(q_e['diseases']) or 'None'}")
    st.write(f"**Drugs/Chemicals:** {', '.join(q_e['drugs']) or 'None'}")

    st.write("[Entities] From your answer:")
    st.write(f"**Diseases:** {', '.join(a_e['diseases']) or 'None'}")
    st.write(f"**Drugs/Chemicals:** {', '.join(a_e['drugs']) or 'None'}")
