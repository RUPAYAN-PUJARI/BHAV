import ssl
import requests
from flask_cors import CORS
from flask import Flask, request, jsonify
from gtts import gTTS
import base64
import os
from io import BytesIO
from groq import Groq  # type: ignore
from mtranslate import translate  # type: ignore
import re  # Added for text cleaning

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

#Get API key from environment

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")

#Debug log - remove after confirming on Render
print("DEBUG: Loaded GROQ_API_KEY from environment:", GROQ_API_KEY[:5] + "***")

#Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

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

    # Interact with Groq API
    messages = [
        {"role": "system", "content": "A helpful polite assistant. Your name is BHAV. When asked about your name, you should say that it is 'BHAV'."},
        {"role": "user", "content": translated_prompt}
    ]
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    assistant_response = response.choices[0].message.content

    # Translate response back to Bengali
    translated_response = custom_translate(assistant_response, 'bn', 'en')
    call_number = None

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

@app.route("/ping")
def ping():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)