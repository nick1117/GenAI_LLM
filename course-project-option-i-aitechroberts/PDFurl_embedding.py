import os
import requests
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, TextStringObject


pdf_urls = [
    "https://arxiv.org/pdf/2411.02345",
    "https://arxiv.org/pdf/2411.07177",
    "https://arxiv.org/pdf/1706.03762",
    "https://arxiv.org/pdf/2501.03526",
    "https://arxiv.org/pdf/2501.09008",
    "https://arxiv.org/pdf/2501.11196",
    "https://arxiv.org/pdf/2501.11258",
    "https://arxiv.org/pdf/2501.14678",
    "https://arxiv.org/pdf/2501.15994",
    "https://arxiv.org/pdf/2501.18011",
    "https://arxiv.org/pdf/2502.04367",
    "https://arxiv.org/pdf/2502.05517",
    "https://arxiv.org/pdf/2502.06171",
    "https://arxiv.org/pdf/2502.07836",
    "https://arxiv.org/pdf/2502.09686",
    "https://arxiv.org/pdf/2502.09805",
    "https://arxiv.org/pdf/2502.09813",
    "https://arxiv.org/pdf/2502.10547",
    "https://arxiv.org/pdf/2502.11200",
    "https://arxiv.org/pdf/cs/0411025",
]

folder_name = "url_documents"
os.makedirs(folder_name, exist_ok=True)

def download_pdf(url, output_path):
    """Download PDF from URL and save it to the specified path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"(Success) Downloaded: {os.path.abspath(output_path)}")
    else:
        print(f"(Failed) Failed to download {url}")

def embed_metadata(pdf_path, output_path, source_url):
    """Embed source URL into the metadata of a PDF."""
    try:
        reader = PdfReader(pdf_path)

        # Check if the PDF is encrypted
        if reader.is_encrypted:
            print(f"(Warning) Skipping {pdf_path} - PDF is encrypted.")
            return

        writer = PdfWriter()

        for page in reader.pages:
            writer.add_page(page)

        metadata = reader.metadata or {}
        new_metadata = {NameObject(key): TextStringObject(str(value)) for key, value in metadata.items()}

        new_metadata[NameObject("/SourceURL")] = TextStringObject(source_url)

        writer.add_metadata(new_metadata)

        with open(output_path, "wb") as file:
            writer.write(file)

        print(f"(Complete) Metadata added and saved in: {os.path.abspath(output_path)}")
    
    except Exception as e:
        print(f"(ERROR) Error processing {pdf_path}: {e}")

def process_pdfs(pdf_urls):
    """Process multiple PDFs: download, embed metadata, and clean up."""
    for url in pdf_urls:
        filename = url.split("/")[-1] 
        temp_pdf_filename = f"{filename}.pdf" 
        pdf_with_metadata_filename = os.path.join(folder_name, f"{filename}.pdf")

        download_pdf(url, temp_pdf_filename)

        # Check if the downloaded PDF is valid before processing
        try:
            PdfReader(temp_pdf_filename)  # Try to read the file
            embed_metadata(temp_pdf_filename, pdf_with_metadata_filename, url)
        except Exception as e:
            print(f"(WARNING) Skipping {temp_pdf_filename} - Unable to process: {e}")

        os.remove(temp_pdf_filename)  # Remove temporary file
        print(f"Temporary file removed: {temp_pdf_filename}")

process_pdfs(pdf_urls)
