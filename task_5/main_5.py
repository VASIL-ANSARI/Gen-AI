import streamlit as st
from helper import create_vector_store, retrieve_context
from sentiment_analyzer import MedicalSentimentAnalyzer
from anxiety_detector import detect_medical_anxiety
from datetime import datetime
import json
from pathlib import Path

st.set_page_config(page_title="Sentiment-Medical Q&A Chatbot")

st.title("🏥 Sentiment - Medical Q&A Chatbot (MedQuAD)")
st.warning("⚠️ This chatbot is for educational purposes only.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_json" not in st.session_state:
    st.session_state.chat_json = "[]"

folder_path = Path("vector_db_store")

if folder_path.exists() and folder_path.is_dir():
    st.session_state.db_ready = True

sentiment_analyzer = MedicalSentimentAnalyzer()

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_store()
    st.session_state.db_ready = True
    st.success("Knowledge Database Created")


def sentiment_to_numeric(label):
    mapping = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }
    return mapping.get(label.lower(), 0)

def build_sentiment_trend(chat_history):
    turns = []
    values = []
    confidence = []

    for i, chat in enumerate(chat_history, start=1):
        turns.append(i)
        values.append(sentiment_to_numeric(chat["sentiment"]))
        confidence.append(chat["sentiment_score"])

    return turns, values, confidence

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
        # render_sentiment(sentiment_data)

        # 2. detect anxiety of the user
        anxiety_flag = detect_medical_anxiety(question, sentiment_data)

        if anxiety_flag:
            st.warning(
                "⚠️ You seem concerned. I’ll respond with extra care and clarity. "
                "This is not medical advice."
            )

        # 3. RAG response
        rag = retrieve_context(sentiment, anxiety_flag)


        # 4. Get context from previous questions of the session
        chat_history = st.session_state.chat_history

        previous_answer = " ".join(
                    f"{c['answer']}"
                    for c in chat_history[-1:]
                )
        
        question_to_ask = previous_answer + ". " + question

        print(question_to_ask)
        response = rag.invoke(question_to_ask)

        # 5. Store interaction
        st.session_state.chat_history.append({
            "question": question,
            "answer": response,
            "sentiment": sentiment,
            "sentiment_score": sentiment_data["scores"][sentiment.lower()],
            "anxiety": anxiety_flag,
            "timestamp": datetime.now().isoformat()
        })

        st.session_state.chat_json = json.dumps(
            st.session_state.chat_history,
            indent=4,
            ensure_ascii=False
        )

    # st.metric(
    #     label="Emotional State",
    #     value=sentiment_data["label"],
    #     delta=f"{sentiment_data['scores'][sentiment_data["label"].lower()] * 100:.0f}% confidence"
    # )

st.subheader("🗂️ Chat History")

for i, chat in enumerate(st.session_state.chat_history):
    st.markdown(f"User: {chat['question']}")
    st.markdown(f"Bot: {chat['answer']}")
    st.caption(
        f"Sentiment: {chat['sentiment']} | "
        f"Anxiety Detected: {'Yes' if chat['anxiety'] else 'No'}"
    )
    st.markdown("---")

clear_history_btn = st.button("🗑️ Clear Chat History")

if clear_history_btn:
    st.session_state.chat_history = []
    st.session_state.chat_json = "[]"
    st.success("Chat history and session JSON cleared.")
    st.experimental_rerun()


anxiety_turns = [
    i + 1
    for i, chat in enumerate(st.session_state.chat_history)
    if chat["anxiety"]
]

if anxiety_turns:
    st.warning(
        f"⚠️ Anxiety detected at turns: {', '.join(map(str, anxiety_turns))}"
    )

if len(st.session_state.chat_history) >= 2:
    st.subheader("📈 Sentiment Trend Over Conversation")

    turns, sentiment_values, confidence_scores = build_sentiment_trend(
        st.session_state.chat_history
    )

    st.line_chart(
        {
            "Sentiment Polarity": sentiment_values,
            "Confidence": confidence_scores
        }
    )

    st.caption(
        "Sentiment Polarity: -1 = Negative, 0 = Neutral, +1 = Positive"
    )


with st.expander("🔍 View Raw Session JSON"):
    st.json(json.loads(st.session_state.chat_json))

if st.session_state.chat_history:
    st.download_button(
        label="📥 Download Chat History (JSON)",
        data=st.session_state.chat_json,
        file_name="medical_chat_session.json",
        mime="application/json"
    )