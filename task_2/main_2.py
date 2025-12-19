import streamlit as st
from PIL import Image

from langchain_helper_two import create_vector_db, get_gemini_response

st.set_page_config(page_title="ElevanceSkills Multi Modal Chatbot")

# Manual ingestion button
if st.button("🔄 Run ingestion now"):
    create_vector_db()
    st.success("Knowledge Database Updated")

st.title("ElevanceSkills — Multi Model Chatbot")
    
# UI: query box
input = st.text_input("Ask Question: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

if st.button("Ask"):
    with st.spinner("Answering..."):
        response = get_gemini_response(input, image)
        # answer = rag.invoke(query)


    st.subheader("Answer")
    st.write(response["text"])
    if response["image"]:
        st.image(response["image"], caption="Generated Explanation")