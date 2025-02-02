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
import threading

# Download VADER lexicon if not already available
nltk.download('vader_lexicon')

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

# Load config
with open("config.json") as config_file:
    config_data = json.load(config_file)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Global variable to track the current audio playback
current_audio_thread = None

def stop_current_audio():
    global current_audio_thread
    if current_audio_thread and current_audio_thread.is_alive():
        current_audio_thread.do_run = False
        current_audio_thread.join()


def play_audio(audio_data):
    global current_audio_thread
    stop_current_audio()
    
    def audio_thread():
        t = threading.currentThread()
        with BytesIO(audio_data) as audio_fp:
            audio_fp.seek(0)
            data, fs = sd.rec(None, samplerate=24000, channels=1, dtype='int16')
            if getattr(t, "do_run", True):
                sd.play(data, fs)
                sd.wait()
    
    current_audio_thread = threading.Thread(target=audio_thread)
    current_audio_thread.start()


def custom_translate(text, to_lang="en", from_lang="bn"):
    return translate(text, to_lang, from_lang)


def clean_text_for_tts(text):
    """
    Cleans the AI response by removing unwanted symbols, extra spaces, and special characters
    that should not be read aloud.
    """
    text = text.replace('*', '')  # Remove stars
    text = re.sub(r'[(){}<>]', '', text)  # Remove brackets and similar symbols
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("prompt", "")
    
    translated_prompt = custom_translate(user_prompt, 'en', 'bn')
    
    sentiment_score = sia.polarity_scores(translated_prompt)["compound"]
    is_negative = sentiment_score < -0.5  
    
    messages = [
        {"role": "system", "content": "A helpful polite assistant."},
        {"role": "user", "content": translated_prompt}
    ]
    response = client.chat.completions.create(model="llama-3.2-3b-preview", messages=messages)
    assistant_response = response.choices[0].message.content
    
    translated_response = custom_translate(assistant_response, 'bn', 'en')

    call_number = None
    if is_negative:
        translated_response += "\n\n(সতর্কতা জরুরি নম্বর)"
        call_number = "1098"
    
    cleaned_response = clean_text_for_tts(translated_response)
    
    audio_fp = BytesIO()
    tts = gTTS(text=cleaned_response, lang='bn', slow=False, tld="com")
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    
    audio_data = audio_fp.read()
    play_audio(audio_data)  # Ensure only one audio is played at a time
    
    audio_base64 = base64.b64encode(audio_data).decode()
    
    return jsonify({"response": translated_response, "audio": audio_base64, "call": call_number})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
