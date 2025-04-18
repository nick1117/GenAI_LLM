import streamlit as st
from PIL import Image
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
import os
import base64
from io import BytesIO

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./nih-cl-cm500-nchermak-2ba6-c06ffadeffcb.json" 
PROJECT_ID = "nih-cl-cm500-nchermak-2ba6"
REGION = "us-central1"

# cant upload image from local disk?
# convert image to Base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

chat_vision_model = ChatVertexAI(model_name="gemini-1.0-pro-vision")

st.title("Image Inquiry Chatbot")
st.write("Upload an image and ask a question about it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
user_query = st.text_area("Enter your question about the image")

if uploaded_file and user_query:

    image = Image.open(uploaded_file)
    #convert to b64
    image_b64 = image_to_base64(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_message_part = {
        "type": "media",
        "mime_type": "image/jpeg",
        "data": image_b64  #change dor image b64
    }

    message = HumanMessage(content=[image_message_part, user_query])

    with st.spinner("Processing..."):
        output = chat_vision_model.invoke([message])

    st.subheader("Response:")
    st.write(output.content)
