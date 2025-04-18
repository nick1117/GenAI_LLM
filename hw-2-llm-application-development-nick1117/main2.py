import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import requests


# need to do this 
# pip install -r requirements.txt  - poetry
# then use: streamlit run main.py
# acutally use python -m streamlit run main.py (streamlit run main.py doesnt work)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./nih-cl-cm500-nchermak-2ba6-c06ffadeffcb.json" # place the key JSON file in the same folder as your notebook

PROJECT_ID = "nih-cl-cm500-nchermak-2ba6" # use your project id, can get from key.json
REGION = "us-central1"  #


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
    "What has been the highlight or challange of your day?",
    "If you could do anything right now, what would it be?"
]

user_querys = []

for question in questions:
    user_query = st.text_input(question, "")
    user_querys.append(user_query)


if all(user_querys):
    prompt_template_name = PromptTemplate(
        input_variables=['user_input'],
        # template="""
        # You are an AI assistant that classifies users' moods based on their responses.
        # Possible moods: Happy, Sad, Stressed, Excited, Neutral

        # The user inquiry: "{user_input}"

        # Classify their mood:
        # """
        template="""
        You are an AI assistant that classifies the user's mood based on their responses and then determines the most relevant news category.
        
        Possible moods: Happy, Sad, Stressed, Excited, Neutral.
        Possible news categories: Comedy, Sports, Technology, Business, Politics.
        
        User responses:
        {user_input}
        
        Step 1: Classify the user's mood.
        Step 2: Choose the most relevant news category for the user based on their mood.
        Step 3: Respond **only** with the name of the selected news category. Do not provide any additional text.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template_name)
    #category = chain.invoke({"user_input": user_query})
    # need to make lower case
    category = chain.invoke({"user_input": "\n".join(user_query)})['text'].strip().lower()


    def fetch_news(category):
        api_key = "7a2968502158483ba75e569b4b003ca3"
        url = f"https://newsapi.org/v2/top-headlines?category={category}&country=us&apiKey={api_key}"
        response = requests.get(url).json()
        articles = response.get("articles", [])[:3]
        return "\n".join([f"- {a.get('title', 'No title')}: {a.get('description', 'No description')}" for a in articles])

    news_summary = fetch_news(category)
    st.write(f"News Category: {category}")
    st.write(news_summary)

    #st.write(f"You should contact: {result['text']}")
    
# News API Key: 7a2968502158483ba75e569b4b003ca3

