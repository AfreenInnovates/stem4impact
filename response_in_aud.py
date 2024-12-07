import streamlit as st
import google.generativeai as gen_ai
from PIL import Image
import os
from gtts import gTTS
import base64
import re
from datetime import datetime

# Configure Gemini API
gen_ai.configure(api_key="")
gemini = gen_ai.GenerativeModel("gemini-1.5-flash")

def upload_image_to_gemini(image_path):
    uploaded_file = gen_ai.upload_file(path=image_path, display_name=os.path.basename(image_path))
    return uploaded_file

# Helper functions for text to speech and cleaning markdown
def strip_markdown(text):
    """
    Removes Markdown syntax, leaving plain text.
    """
    text = re.sub(r"[*_`~]", "", text)
    return text

def text_to_speech(text, autoplay=True):
    """
    Converts text to speech and displays an audio player.
    Args:
        text (str): Text to convert to speech
        autoplay (bool): Whether to autoplay the audio
    """
    plain_text = strip_markdown(text)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = f"audio_{timestamp}.mp3"
    
    # Generate audio
    tts = gTTS(text=plain_text, lang='en')
    tts.save(audio_file_path)
    
    # Read audio content
    with open(audio_file_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    
    # Convert to base64 for embedding
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Display audio player
    st.markdown(f"""
        <audio autoplay="{str(autoplay).lower()}" controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

# Streamlit UI
st.title("📸 Gemini Pro - Image ChatBot")

# Initialize chat session and history in Streamlit session state
if "chat_session" not in st.session_state:
    st.session_state.chat_session = gemini.start_chat(history=[])
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "image_description_done" not in st.session_state:
    st.session_state.image_description_done = False  # Track if the image description step is completed

# Step 1: Select Input Method
st.subheader("Step 1: Choose Image Input Method")
input_method = st.radio("Select how you want to provide an image:", ("Upload Image", "Capture Image"))

uploaded_image = None
captured_image = None
image_path = None

# Handle user selection
if input_method == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        # Save uploaded image to a temporary path
        temp_path = f"temp_uploaded_image.{uploaded_image.name.split('.')[-1]}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        image_path = temp_path
elif input_method == "Capture Image":
    captured_image = st.camera_input("Capture an image using your camera")
    if captured_image:
        # Save captured image to a temporary path
        image_path = "temp_captured_image.png"
        with open(image_path, "wb") as f:
            f.write(captured_image.getbuffer())

# Process the image
if image_path:
    # Display the image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded/Captured Image", use_column_width=True)

    # Step 2: Upload to Gemini
    st.subheader("Step 2: Uploading the Image to Gemini")
    with st.spinner("Uploading the image..."):
        gemini_file = upload_image_to_gemini(image_path)
    st.success(f"Image uploaded successfully as: {gemini_file.uri}")

    # Step 3: Automatically Describe the Image (only once)
    if not st.session_state.image_description_done:
        st.subheader("Step 3: Gemini Describes the Image")
        with st.spinner("Generating a description..."):
            description_response = gemini.generate_content([gemini_file, "What is in the image?"])
        description_text = description_response.text
        st.session_state.chat_history.append(("assistant", description_text))  # Save to chat history
        st.session_state.image_description_done = True  # Mark description as completed
        st.write("**Image Description:**")
        st.markdown(description_text)

        # Convert Gemini's description to speech
        text_to_speech(description_text, autoplay=False)

    # Step 4: Chat about the Image
    st.subheader("Step 4: Chat about the Image")

    # Display the chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

            # Convert assistant's responses to speech
            if role == "assistant":
                text_to_speech(message, autoplay=False)

    # User input for questions about the image
    user_prompt = st.chat_input("Ask Gemini-Pro about the image...")
    if user_prompt:
        # Add user's message to chat history and display it
        st.session_state.chat_history.append(("user", user_prompt))
        st.chat_message("user").markdown(user_prompt)

        # Send message to Gemini with the image and user's question
        gemini_response = st.session_state.chat_session.send_message([gemini_file, user_prompt])
        response_text = gemini_response.text

        # Add Gemini's response to chat history and display it
        st.session_state.chat_history.append(("assistant", response_text))
        with st.chat_message("assistant"):
            st.markdown(response_text)

            # Convert Gemini's response to speech
            text_to_speech(response_text, autoplay=True)
else:
    st.info("Please select and provide an image to proceed.")

# Clean up temporary files
if image_path:
    os.remove(image_path)
