FROM python:3.10
WORKDIR /app
COPY /url_documents /url_documents
COPY nih_key.json nih_key.json
COPY .env .env
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY app.py app.py
EXPOSE 8080
CMD streamlit run --server.port 8080 --server.enableCORS false app.py 