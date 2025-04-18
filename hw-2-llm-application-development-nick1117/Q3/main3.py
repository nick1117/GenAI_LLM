import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./key.json"

PROJECT_ID = "nih-cl-cm500-ysakhale-e2a6"
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

def get_weather_forecast(location="Pittsburgh", days=4):
    url = f"https://api.weatherbit.io/v2.0/forecast/daily?city={location}&key=594f1a8f78b74bc78328c20008adf0a9&days={days}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Could not fetch forecast data"}

st.title("Weather Assistant")
st.write("Ask about the current and upcoming weather")

query = st.text_input("Ask a weather-related question:")

if query:
    if "today" in query.lower():
        forecast_data = get_weather_forecast(days=1)
        if "error" not in forecast_data:
            today_weather = forecast_data["data"][0]
            temp = today_weather["temp"]
            condition = today_weather["weather"]["description"]
            st.write(f"The current temperature in Pittsburgh is {temp}°C with {condition}.")
        else:
            st.write("Error fetching current weather.")
    elif "next" in query.lower() or "forecast" in query.lower():
        forecast_data = get_weather_forecast()
        if "error" not in forecast_data:
            st.subheader("4-Day Weather Forecast:")
            for day in forecast_data["data"][:4]:
                date = day["valid_date"]
                temp = day["temp"]
                condition = day["weather"]["description"]
                st.write(f"On {date}, the temperature will be {temp}°C with {condition}.")
        else:
            st.write("Error fetching forecast.")
    else:
        prompt = PromptTemplate(
            input_variables=['query'],
            template="""
            You are a weather assistant. Answer questions related to weather conditions.
            
            User query: {query}
            """
        )
        
        weather_chain = LLMChain(llm=llm, prompt=prompt)
        weather_result = weather_chain.invoke({"query": query})
        st.write(weather_result["text"].strip())


