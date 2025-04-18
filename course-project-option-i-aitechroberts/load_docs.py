# file: load_data.py

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(pdf_directory: str):
    docs = []
    # Use a text splitter that splits by chunk size (e.g., 1,000 tokens or 1,000 characters).
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(pdf_directory, filename))
            pdf_docs = loader.load()
            # Split into smaller chunks
            for d in pdf_docs:
                splits = text_splitter.split_text(d.page_content)
                for s in splits:
                    # Create 'Document' objects that LangChain expects
                    docs.append({
                        "page_content": s,
                        "metadata": {
                            "source": filename
                        }
                    })
    return docs

if __name__ == "__main__":
    pdf_directory = "data/papers"
    documents = load_and_chunk_pdfs(pdf_directory)
    print(f"Loaded {len(documents)} chunks from PDF files.")

# OR

import os
import glob
from langchain.document_loaders import PyPDFLoader

def load_papers(folder_path: str):
    docs = []
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        loader = PyPDFLoader(file_path)
        loaded_docs = loader.load()
        docs.extend(loaded_docs)
    return docs

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

raw_docs = load_papers("./papers")
split_docs = []
for doc in raw_docs:
    for chunk in text_splitter.split_text(doc.page_content):
        metadata = {"source": doc.metadata.get("source", ""), 
                    "title": doc.metadata.get("title", ""), 
                    # Add domain metadata if you know it
                    "domain": "biology"  # or parse it from the filename
                   }
        split_docs.append(Document(page_content=chunk, metadata=metadata))
