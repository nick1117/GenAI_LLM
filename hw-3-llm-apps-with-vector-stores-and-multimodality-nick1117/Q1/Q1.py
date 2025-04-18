from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import glob
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage, AIMessage


# Install dependencies
# pip install -r requirements.txt - poetry
# Run using: python -m streamlit run main.py

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./nih-cl-cm500-nchermak-2ba6-c06ffadeffcb.json" 
PROJECT_ID = "nih-cl-cm500-nchermak-2ba6"
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

llm = VertexAI(
    model_name="gemini-1.5-pro-001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

EMBEDDING_MODEL_NAME = "text-embedding-004"
embedding_model = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_vector_store(pdf_path):
    all_chunks = []
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    all_chunks.extend(chunks)

    # Create vector store
    vector_store = Chroma(
        persist_directory="db",
        embedding_function=embedding_model,
        collection_name=f"lecture_{os.path.basename(pdf_path)}"
    )

    if all_chunks:
        vector_store.add_documents(all_chunks)
        st.write(f"Added {len(all_chunks)} document chunks from {os.path.basename(pdf_path)} to the vector store.")
    
    return vector_store

st.title("Lecture Q&A chatbot")

pdf_files = sorted(glob.glob(os.path.join("PDFs", "*.pdf")))[:6]


if not pdf_files:
    st.error("No lecture PDFs found. Ensure they are placed in the 'PDFs/' directory.")

selected_pdf = st.selectbox("Choose a lecture PDF:", pdf_files)

vector_store = load_vector_store(selected_pdf)

## chat history: 

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        SystemMessage("You are an assistant that answers questions based on lecture content.")
    )

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_prompt = st.chat_input("Ask a question based on the lecture PDF:")

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append(HumanMessage(user_prompt))
    
    #vector store info
    retriever_mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    retriever_similarity = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    retrieved_docs_mmr = retriever_mmr.get_relevant_documents(user_prompt)
    retrieved_docs_similarity = retriever_similarity.get_relevant_documents(user_prompt)

    docs_text_mmr = "\n\n".join(d.page_content for d in retrieved_docs_mmr)
    docs_text_similarity = "\n\n".join(d.page_content for d in retrieved_docs_similarity)

    #prompt template
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "You are an assistant that answers questions based on lecture content. "
            "Provide a concise but informative response. "
            "If the answer is not in the provided context, say 'I don't know'.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}"
        )
    )

    llm = VertexAI(model_name="gemini-1.5-pro-002", temperature=0.3, verbose=True)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    #get responses from both retrieval methods
    chain_input_mmr = {"question": user_prompt, "context": docs_text_mmr}
    chain_input_similarity = {"question": user_prompt, "context": docs_text_similarity}

    answer_mmr = llm_chain.run(chain_input_mmr)
    answer_similarity = llm_chain.run(chain_input_similarity)

    #display and compare
    with st.chat_message("assistant"):
        st.subheader("MMR Retriever Answer:")
        st.markdown(answer_mmr)

        st.subheader("Similarity Search Answer:")
        st.markdown(answer_similarity)

    st.session_state.messages.append(AIMessage(f"MMR Answer: {answer_mmr}\n\nSimilarity Answer: {answer_similarity}"))


