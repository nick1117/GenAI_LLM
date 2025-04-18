from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
from langchain.memory import ConversationBufferMemory

# Install dependencies
# pip install -r requirements.txt - poetry
# Run using: python -m streamlit run main.py

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./nih-cl-cm500-nchermak-2ba6-c06ffadeffcb.json"  # Replace with your key JSON file
PROJECT_ID = "nih-cl-cm500-nchermak-2ba6"  # Use your Google Cloud project ID
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

st.title("Emotion Detector")
st.write("Answer a few questions, and I'll recommend relevant news based on your mood.")

questions = [
    "Describe your day in one sentence: ",
    "What has been the highlight or challenge of your day?",
    "If you could do anything right now, what would it be?"
]

user_querys = []

for question in questions:
    user_query = st.text_input(question, "")
    user_querys.append(user_query)

def fetch_news_tool(category):
    api_key = "7a2968502158483ba75e569b4b003ca3"
    url = f"https://newsapi.org/v2/top-headlines?category={category}&country=us&apiKey={api_key}"
    response = requests.get(url).json()
    articles = response.get("articles", [])[:3]
    return "\n".join([f"- {a.get('title', 'No title')}: {a.get('description', 'No description')}" for a in articles])

news_tool = Tool(
    name="Fetch News",
    func=fetch_news_tool,
    description="Fetches recent news based on a given category (comedy, sports, technology, business, politics)."
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=[news_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

if all(user_querys):
    user_text = "\n".join(user_querys)

    prompt = f"""
        You are an AI assistant that classifies the user's mood based on their responses and then determines the most relevant news category.
        
        Possible moods: Happy, Sad, Stressed, Excited, Neutral.
        Possible news categories: Comedy, Sports, Technology, Business, Politics.
        
        User responses:
        {user_text}
        
        Step 1: Classify the user's mood.
        Step 2: Choose the most relevant news category for the user based on their mood.
        Step 3: Respond **only** with the name of the selected news category. Do not provide any additional text.
        """

    category = agent.run(prompt).strip().lower()

    news_summary = fetch_news_tool(category)

    st.write(f"### News in {category.capitalize()}")

    #Need to format out of json
    news_articles = news_summary.split("\n")
    for article in news_articles:
        if article.startswith("- "):
            title, description = article[2:].split(":", 1) if ":" in article else (article[2:], "")
            st.markdown(f"**{title.strip()}**")
            st.write(description.strip())
            st.write("---")

