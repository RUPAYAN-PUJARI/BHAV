import ssl
import warnings
import requests
from flask_cors import CORS
from flask import Flask, request, jsonify
from gtts import gTTS
import base64
import json
import os
import sounddevice as sd  # type: ignore
import speech_recognition as sr
from io import BytesIO
from groq import Groq  # type: ignore
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from mtranslate import translate  # type: ignore
import re  # Added for text cleaning

# Download VADER lexicon if not already available
nltk.download('vader_lexicon')

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")

# Debug log - remove after confirming on Render
print("DEBUG: Loaded GROQ_API_KEY from environment:", GROQ_API_KEY[:5] + "*****")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def custom_translate(text, to_lang="en", from_lang="bn"):
    return translate(text, to_lang, from_lang)

def clean_text_for_tts(text):
    """
    Cleans the AI response by removing unwanted symbols, extra spaces, and special characters
    that should not be read aloud.
    """
    text = text.replace('*', '')  # Remove stars
    text = re.sub(r'[=\\/]', '', text)
    text = re.sub(r'[(){}<>]', '', text)  # Remove brackets and similar symbols
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Endpoint to handle chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("prompt", "")
    
    # Translate the user's message to English if in Bengali
    translated_prompt = custom_translate(user_prompt, 'en', 'bn')
    
    # Analyze sentiment
    sentiment_score = sia.polarity_scores(translated_prompt)["compound"]
    print(f"Sentiment Analysis Score: {sentiment_score}")
    is_negative = sentiment_score < -0.5  # Threshold for negativity
    
    # Interact with Groq API
    messages = [
        {"role": "system", "content": "A helpful polite assistant."},
        {"role": "user", "content": translated_prompt}
    ]
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    assistant_response = response.choices[0].message.content
    
    # Translate response back to Bengali
    translated_response = custom_translate(assistant_response, 'bn', 'en')

    # Modify response if sentiment is negative
    call_number = None
    if is_negative:
        translated_response += "\n\n(সতর্কতা জরুরি নম্বর)"
        call_number = "1098"  # Example emergency number
    
    # Clean the response before converting to speech
    cleaned_response = clean_text_for_tts(translated_response)

    # Convert response to speech in Bengali
    audio_fp = BytesIO()
    tts = gTTS(text=cleaned_response, lang='bn', slow=False, tld="com")
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    
    # Encode audio to base64
    audio_base64 = base64.b64encode(audio_fp.read()).decode()
    
    return jsonify({"response": translated_response, "audio": audio_base64, "call": call_number})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
