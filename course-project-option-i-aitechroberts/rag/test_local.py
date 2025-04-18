from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


pdf_path = "rag\Testpaper.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_pages = splitter.split_documents(pages)

print(f"Loaded {len(splitted_pages)} chunks from the PDF")

# uncomment for cloud??
# embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

# uncomment for local
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = embedding_model.embed_documents([chunk.page_content for chunk in splitted_pages])

print(f"Generated {len(embeddings)} embeddings")

vector_store = Chroma(collection_name="pdf_summaries",
                      embedding_function=VertexAIEmbeddings(model_name="text-embedding-004"))

chroma_docs = vector_store.from_documents(documents=splitted_pages, 
                                          embedding=VertexAIEmbeddings(model_name="text-embedding-004"))

print("Stored embeddings in ChromaDB")

query = "Explain the key ideas in this paper?"
similar_vectors = chroma_docs.similarity_search(query, k=3)

print("Retrieved relevant sections:")
for doc in similar_vectors:
    print(f"- {doc.page_content[:500]}...\n")

retriever = chroma_docs.as_retriever(search_type="mmr", k=3)
retrieved_docs = retriever.get_relevant_documents("Summarize the findings of this paper.")

print("MMR Retrieved documents:")
for doc in retrieved_docs:
    print(f"- {doc.page_content[:500]}...\n")  
