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
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

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

def custom_translate(text, to_lang="en", from_lang="bn"):
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={from_lang}&tl={to_lang}&dt=t&q={text}"
    response = requests.get(url, verify=False)  # Disable SSL verification
    return response.json()[0][0][0]

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
        translated_response += "\n\n(ALERTING EMERGENCY NUMBER!!)"
        call_number = "+919875357018"  # Example emergency number
    
    # Convert response to speech in Bengali
    audio_fp = BytesIO()
    tts = gTTS(text=translated_response, lang='bn', slow=False)
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    
    # Encode audio to base64
    audio_base64 = base64.b64encode(audio_fp.read()).decode()
    
    return jsonify({"response": translated_response, "audio": audio_base64, "call": call_number})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
