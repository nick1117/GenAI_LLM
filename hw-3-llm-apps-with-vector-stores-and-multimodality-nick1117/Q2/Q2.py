import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./nih-cl-cm500-nchermak-2ba6-c06ffadeffcb.json" 
PROJECT_ID = "nih-cl-cm500-nchermak-2ba6"
REGION = "us-central1"

#languagues for deep_translator
LANGUAGES = {
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Hindi": "hi"
}

st.title("PDF Translator using Deep Translator")
st.write("Upload a PDF file in English and translate it into any language.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    temp_file_path = "temp_uploaded.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    target_language = st.selectbox("Choose a language for translation:", list(LANGUAGES.keys()))

    with st.spinner("Extracting text from PDF..."):
        pdf_loader = PyPDFLoader(temp_file_path)  # Pass the saved file path
        pages = pdf_loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    translated_chunks = []
    translator = GoogleTranslator(source="en", target=LANGUAGES[target_language])  # Initialize Translator

    with st.spinner(f"Translating to {target_language}..."):
        for chunk in chunks:
            translated_chunk = translator.translate(chunk.page_content)
            translated_chunks.append(translated_chunk)

    translated_text = "\n\n".join(translated_chunks)

    st.subheader(f"Translated Text in {target_language}:")
    st.text_area("", translated_text, height=500)

    os.remove(temp_file_path)
