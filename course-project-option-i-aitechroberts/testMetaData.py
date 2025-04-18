import os
from PyPDF2 import PdfReader

# Path to the folder containing PDFs
folder_path = "url_documents"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        
        pdf_reader = PdfReader(pdf_path)
        metadata = pdf_reader.metadata
        
        print(f"\nMetadata for: {filename}")
        if metadata:
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("No metadata found")
