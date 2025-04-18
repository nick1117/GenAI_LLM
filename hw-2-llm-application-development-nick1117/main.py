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

st.title("XYZ Dealership")
st.write("Enter your inquiry below, and I'll direct you to the correct department!")

user_query = st.text_input("What can I help you with?")

if user_query:
    prompt_template_name = PromptTemplate(
        input_variables=['user_input'],
        template="""
        You are an assistant for XYZ Dealership. Classify the user's inquiry into one of the following departments:
        - Sales
        - Car Service - Emergency
        - Car Service - Regular Maintenance
        - Human Resources
        - Other

        The user inquiry: "{user_input}"

        Respond **only** with the department name. Do not provide any additional text or explanation.
        """

    )
    chain = LLMChain(llm=llm, prompt=prompt_template_name)
    result = chain.invoke({"user_input": user_query})
    st.write(f"You should contact: {result['text']}")

