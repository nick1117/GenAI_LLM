[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bWn0IfCm)


Question 1: 

run "python -m streamlit run Q1.py" in the Q1 folder path. Then select the drop down option to choose which lecture PDF to ask and then ask your question.

Details on how the lecture Q&A chat bot was built: 

1. Setup and imported dependencies
2. Setup authentication and initialized vertex AI
3. Loaded and processed lecture PDFs
   - Use PyPDFLoader to read PDF and then splot text into chucks with RecursiveCharacterTextSplitter.
4. Created a chroma vector store to store and retrieve embeddings
5. Prompt user to select PDF and create a vector store for that PDF
6. Implement chat history and prompt user to ask a question
7. Implement document retrieval (MMR and Similiarity search)
8. Used vector_store.as_retriever() with k=2 (retrieves top 2 relevant documents)
9. Defined LLM Prompt and query execution
10. Created a PromptTemplate to structure model response and use LLChain/VertexAIA to generate answers
11. Run query through both retrevial methods
12. Display results in streamlit

MMR vs Similarity Search Comparison:
MMR balances relevance and diversity by selecting document chunks that provide the most new information while avoiding redundancy. Similarity search, on the other hand, retrieves the most relevant chunks based purely on cosine similarity, which can lead to repetitive information. MMR is better for ambiguous or multi-faceted queries, as it ensures a broader range of perspectives from the document. However, similarity search is more useful when precision is required, such as retrieving a specific definition or exact match from the text. So in conclusion, MMR is better for this application. 