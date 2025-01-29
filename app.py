import ssl
import warnings
import requests
from flask import Flask, request, jsonify
from gtts import gTTS
import base64
import json
import os
from mtranslate import translate # type: ignore
import sounddevice as sd # type: ignore
import speech_recognition as sr
from io import BytesIO
from groq import Groq # type: ignore

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Load config
with open("config.json") as config_file:
    config_data = json.load(config_file)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

# Endpoint to handle chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("prompt", "")
    
    # Translate the user's message to English if in Bengali
    translated_prompt = translate(user_prompt, 'en', 'bn')

    # Interact with Groq API
    messages = [{"role": "system", "content": "A helpful polite assistant."},
                {"role": "user", "content": translated_prompt}]
    response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=messages)
    assistant_response = response.choices[0].message.content

    # Translate response back to Bengali
    translated_response = translate(assistant_response, 'bn', 'en')

    # Convert response to speech in Bengali
    audio_fp = BytesIO()
    tts = gTTS(text=translated_response, lang='bn', slow=False)
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)

    # Encode audio to base64
    audio_base64 = base64.b64encode(audio_fp.read()).decode()

    return jsonify({"response": translated_response, "audio": audio_base64})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
