import streamlit as st
from helper import create_vector_store, retrieve_context
from sentiment_analyzer import MedicalSentimentAnalyzer
from anxiety_detector import detect_medical_anxiety
from datetime import datetime
import json

st.set_page_config(page_title="Sentiment-Medical Q&A Chatbot")

st.title("🏥 Sentiment - Medical Q&A Chatbot (MedQuAD)")
st.warning("⚠️ This chatbot is for educational purposes only.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

sentiment_analyzer = MedicalSentimentAnalyzer()

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_store()
    st.session_state.db_ready = True
    st.success("Knowledge Database Created")


def render_sentiment(sentiment_data):
    sentiment = sentiment_data["label"].lower()
    confidence = sentiment_data["scores"].get(sentiment, 0.0)

    if sentiment == "positive":
        st.success("🙂 Positive sentiment detected")
    elif sentiment == "negative":
        st.error("😟 Negative sentiment detected")
    else:
        st.info("😐 Neutral sentiment detected")

    st.progress(confidence)
    st.caption(f"Confidence score: {confidence:.2f}")


question = st.text_input("Ask a question:")

if st.button("Ask") and question:
    if not st.session_state.get("db_ready", False):
        st.error("Please create the knowledge base first.")
        st.stop()
        
    with st.spinner("Analyzing your question..."):

        # 1. Sentiment detection
        sentiment_data = sentiment_analyzer.analyze(question)
        sentiment = sentiment_data['label']

        # sentiment_data = detect_sentiment(question)
        render_sentiment(sentiment_data)

        # 2. detect anxiety of the user
        anxiety_flag = detect_medical_anxiety(question, sentiment_data)

        if anxiety_flag:
            st.warning(
                "⚠️ You seem concerned. I’ll respond with extra care and clarity. "
                "This is not medical advice."
            )

        # 3. RAG response
        rag = retrieve_context(sentiment, anxiety_flag)

        response = rag.invoke(question)

        # 4. Store interaction
        st.session_state.chat_history.append({
            "question": question,
            "answer": response,
            "sentiment": sentiment,
            "anxiety": anxiety_flag,
            "timestamp": datetime.now().isoformat()
        })

    st.metric(
        label="Emotional State",
        value=sentiment_data["label"],
        delta=f"{sentiment_data['scores'][sentiment_data["label"].lower()] * 100:.0f}% confidence"
    )


if st.session_state.chat_history:
    json_data = json.dumps(
        st.session_state.chat_history,
        indent=4,
        ensure_ascii=False
    )

    st.download_button(
        label="📥 Download Chat History (JSON)",
        data=json_data,
        file_name="medical_chat_session.json",
        mime="application/json"
    )


st.subheader("🗂️ Chat History")

for i, chat in enumerate(st.session_state.chat_history):
    st.markdown(f"User: {chat['question']}")
    st.markdown(f"Bot: {chat['answer']}")
    st.caption(
        f"Sentiment: {chat['sentiment']} | "
        f"Anxiety Detected: {'Yes' if chat['anxiety'] else 'No'}"
    )
    st.markdown("---")


if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()


with st.expander("🔍 View Raw Session JSON"):
    st.json(st.session_state.chat_history)