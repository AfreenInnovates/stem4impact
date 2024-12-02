import streamlit as st 
import google.generativeai as genai
from gtts import gTTS
import os
from datetime import datetime
import uuid
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import queue
import threading
import av
import numpy as np
import base64
import re

GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit app title
st.title("Image Q&A")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "sample_file" not in st.session_state:
    st.session_state.sample_file = None 

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None 

if "chat" not in st.session_state:
    st.session_state.chat = None 

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_audio(frame):
    """Process audio frames."""
    return av.AudioFrame.from_ndarray(
        frame.to_ndarray(), 
        layout="mono"
    )

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.transcript_queue = queue.Queue()
        self.is_recording = False

    def recv(self, frame):
        """Receive and process audio frames."""
        if self.is_recording:
            try:
                # Convert audio frame to text
                audio_data = frame.to_ndarray().tobytes()
                audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
                text = self.recognizer.recognize_google(audio)
                self.transcript_queue.put(text)
            except Exception as e:
                pass
        return frame

# Utility function to save files with a safe name
def save_file(file):
    file_id = str(uuid.uuid4())  # Generate a unique ID for each file
    file_path = f"{file_id}_{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# Function to clean Markdown syntax for plain text
def strip_markdown(text):
    """
    Removes Markdown syntax, leaving plain text.
    """
    # Remove Markdown patterns such as **text**, *text*, _text_, `text`
    text = re.sub(r"[*_`~]", "", text)
    return text

# Function to convert text to speech
def text_to_speech(text, autoplay=True):
    """
    Converts text to speech and displays an audio player.
    Args:
        text (str): Text to convert to speech
        autoplay (bool): Whether to autoplay the audio
    """
    # Clean Markdown from the text
    # Remove Markdown patterns
    plain_text = re.sub(r"[*_`~]", "", text)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = f"audio_{timestamp}.mp3"
    
    # Create audio file
    tts = gTTS(text=plain_text, lang='en')
    tts.save(audio_file_path)
    
    # Read audio file
    with open(audio_file_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    
    # Convert to base64 for HTML embedding
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    if autoplay:
        st.markdown(f"""
            <audio autoplay="true" controls>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """, unsafe_allow_html=True)
    
    # Clean up the temporary file
    os.remove(audio_file_path)

# Section to upload or capture an image
st.header("Upload or Capture an Image")
option = st.radio("Select an Option", ["Upload Image", "Capture Image"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload your image file", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_path = save_file(uploaded_file)

        if st.session_state.uploaded_file != file_path:
            st.session_state.uploaded_file = file_path
            sample_file = genai.upload_file(path=file_path, display_name=uploaded_file.name)
            st.session_state.sample_file = sample_file

            st.session_state.analysis_result = gemini.generate_content([sample_file, "What is in the image?"])

            st.session_state.chat = gemini.start_chat(history=[])
            st.session_state.chat_history = []

        st.image(file_path, caption=uploaded_file.name, use_container_width=True)

elif option == "Capture Image":
    camera_image = st.camera_input("Capture an image using your camera")
    if camera_image:
        file_path = save_file(camera_image)

        if st.session_state.uploaded_file != file_path:
            st.session_state.uploaded_file = file_path
            sample_file = genai.upload_file(path=file_path, display_name="captured_image")
            st.session_state.sample_file = sample_file

            st.session_state.analysis_result = gemini.generate_content([sample_file, "What is in the image?"])
            
            st.session_state.chat = gemini.start_chat(history=[])
            st.session_state.chat_history = []

        st.image(file_path, caption="Captured Image", use_container_width=True)

# Analysis result and chat interface
if st.session_state.sample_file:
    st.write("### Analysis Result:")
    st.markdown(st.session_state.analysis_result.text)  
    text_to_speech(st.session_state.analysis_result.text, autoplay=True)

    st.header("Chat with AI about the Image")
    
    for speaker, message in st.session_state.chat_history:
        with st.chat_message(speaker):
            if speaker == "assistant":
                st.markdown(message)  
            else:
                st.write(strip_markdown(message))  
            
            if speaker == "assistant":
                text_to_speech(message, autoplay=False)

    # Add voice input option
    input_method = st.radio("Choose input method:", ["Text", "Voice"])
    
    if input_method == "Voice":
        audio_processor = AudioProcessor()
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=webrtc_streamer.AudioProcessorMode.AUDIO_ONLY,
            audio_processor_factory=lambda: audio_processor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        
        if webrtc_ctx.state.playing:
            audio_processor.is_recording = True
            try:
                transcript = audio_processor.transcript_queue.get_nowait()
                user_input = transcript
            except queue.Empty:
                user_input = None
        else:
            audio_processor.is_recording = False
            user_input = None
    else:
        user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.write(strip_markdown(user_input)) 

        ai_response = st.session_state.chat.send_message([st.session_state.sample_file, user_input])
        st.session_state.chat_history.append(("assistant", ai_response.text))
        with st.chat_message("assistant"):
            st.markdown(ai_response.text)  
            text_to_speech(ai_response.text, autoplay=True)
