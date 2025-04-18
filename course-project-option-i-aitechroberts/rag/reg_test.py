

# arXiv API:
# http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=20

import feedparser
import urllib.parse
import requests
import fitz

query = "reinforcement learning"
encoded_query = urllib.parse.quote(query)
print(encoded_query)
url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=5"

# feed = feedparser.parse(url)
# for entry in feed.entries:
#     print(f"Title: {entry.title}")
#     print(f"PDF: {entry.link.replace('abs', 'pdf')}")
#     print(f"Summary: {entry.summary}\n")

feed = feedparser.parse(url)

if feed.entries:
    entry = feed.entries[0]  # Get the first result
    pdf_url = entry.link.replace("abs", "pdf")  # Get the PDF link

    # Step 2: Download the PDF
    response = requests.get(pdf_url)
    pdf_filename = "paper.pdf"
    with open(pdf_filename, "wb") as f:
        f.write(response.content)

    print(f"Downloaded PDF: {pdf_filename}")

    # Step 3: Extract text from the PDF
    doc = fitz.open(pdf_filename)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")

    print(f"Extracted Text:\n{full_text[:1000]}...")  # Print first 1000 characters
else:
    print("No papers found.")