import streamlit as st
from helper import create_vector_store, retrieve_context
from sentiment import detect_sentiment
from anxiety import detect_medical_anxiety

st.set_page_config(page_title="Sentiment-Medical Q&A Chatbot")

st.title("🏥 Sentiment - Medical Q&A Chatbot (MedQuAD)")
st.warning("⚠️ This chatbot is for educational purposes only.")

btn = st.button("Create Knowledgebase")
if btn:
    print("Loading data...")
    create_vector_store()
    print("Loaded Data Successfully")
    st.success("Knowledge Database Created")


def render_sentiment(sentiment_data):
    sentiment = sentiment_data["sentiment"]
    confidence = sentiment_data["confidence"]

    if sentiment == "positive":
        st.success("🙂 Positive sentiment detected")
    elif sentiment == "negative":
        st.error("😟 Negative sentiment detected")
    else:
        st.info("😐 Neutral sentiment detected")

    st.progress(confidence)
    st.caption(f"Confidence score: {confidence}")


question = st.text_input("Ask a question:")

if st.button("Ask") and question:
    with st.spinner("Analyzing your question..."):

        # 1. Sentiment detection
        sentiment_data = detect_sentiment(question)
        render_sentiment(sentiment_data)

        # 2. detect anxiety of the user
        anxiety_flag = detect_medical_anxiety(question, sentiment_data)

        if anxiety_flag:
            st.warning(
                "⚠️ You seem concerned. I’ll respond with extra care and clarity. "
                "This is not medical advice."
            )

        # 3. RAG response
        rag = retrieve_context(sentiment_data['sentiment'], anxiety_flag)

        response = rag.invoke(question)

    st.subheader("📌 Answer")
    st.write(response)

    st.metric(
        label="Emotional State",
        value=sentiment_data["sentiment"].capitalize(),
        delta=f"{sentiment_data['confidence'] * 100:.0f}% confidence"
    )   