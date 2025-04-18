import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

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

st.title("City Recommender")


budget = st.sidebar.selectbox("Your Weekly Budget is:", ("Less than $1000", "Between $1000 and $2000", "Between $2000 and $5000", "More than $5000"))

if budget:
    prompt_template_name = PromptTemplate(
        input_variables=['budget'],
        template="I want to spend a nice vacation for a week. My budget is {budget}. Suggest a list of 10 cities to visit that would fit this budget. Display the list of cities as a comma separated list. Only display the cities without any explanation or description."
    )
    chain = LLMChain(llm=llm, prompt=prompt_template_name)
    result = chain.invoke(budget)
    st.write(result)
