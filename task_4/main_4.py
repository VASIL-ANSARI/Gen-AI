import streamlit as st
from helper import get_query_chain

st.set_page_config(page_title="arXiv Expert Chatbot", layout="wide")
st.title("📚 arXiv Domain Expert Chatbot")

# ---------------------------
# Domain Selection
# ---------------------------
TOP_10_CATEGORIES = {
    "Computer Science": "cs",
    "Mathematics": "math",
    "Condensed Matter Physics": "cond-mat",
    "Astrophysics": "astro-ph",
    "Physics": "physics",
    "High Energy Physics – Phenomenology": "hep-ph",
    "Quantum Physics": "quant-ph",
    "High Energy Physics – Theory": "hep-th",
    "General Relativity & Quantum Cosmology": "gr-qc",
    "Electrical Engineering & Systems Science": "eess"
}

selected_domain = st.selectbox(
    "Select Research Domain",
    list(TOP_10_CATEGORIES.keys())
)

domain_code = TOP_10_CATEGORIES[selected_domain]

# ---------------------------
# Knowledge Base Creation
# ---------------------------
# if st.button("Update Knowledge Base"):
#     with st.spinner("Building FAISS Knowledge Base..."):
#         create_vector_db(domain_code)
#     st.success(f"Knowledge Base ready for {selected_domain}")

st.divider()

# ---------------------------
# Chat Dashboard
# ---------------------------
st.subheader("💬 Research Conversation")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a research-level question")

if query:
    with st.spinner("Searching arXiv & reasoning..."):
        chain = get_query_chain(domain_code)
        response = chain.invoke(query)

    # rag = get_qa_chain(domain_code, st.session_state.chat_history)
    # response = rag.invoke(query)

    st.session_state.chat_history.append((query, response))

# ---------------------------
# Display History
# ---------------------------
for user, bot in st.session_state.chat_history:
    st.markdown(f"**🧑 User:** {user}")
    st.markdown(f"**🤖 Assistant:** {bot}")
    st.markdown("---")
