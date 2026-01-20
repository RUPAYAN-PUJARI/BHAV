import ssl
import warnings
import requests
from flask_cors import CORS
from flask import Flask, request, jsonify
from gtts import gTTS
import base64
import json
import os
from io import BytesIO
from groq import Groq  # type: ignore
from googletrans import Translator
from langdetect import detect, LangDetectException
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

translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def translate_bn_to_en(text):
    if not text.strip():
        return text
    try:
        return translator.translate(text, src="bn", dest="en").text
    except Exception as e:
        print("BN→EN translation error:", e)
        return text

def translate_en_to_bn(text):
    if not text.strip():
        return text
    try:
        return translator.translate(text, src="en", dest="bn").text
    except Exception as e:
        print("EN→BN translation error:", e)
        return text

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
    user_prompt = data.get("prompt", "").strip()

    # Step 1: Detect language
    detected_lang = detect_language(user_prompt)

    # Step 2: Normalize input to English
    if detected_lang.startswith("bn"):
        model_input = translate_bn_to_en(user_prompt)
    else:
        model_input = user_prompt

    # Step 3: LLM call
    messages = [
        {
            "role": "system",
            "content": (
                "A polite and helpful assistant. "
                "Your final response MUST be in Bengali (Bangla) only. "
                "Do not use English words or sentences."
            )
        },
        {"role": "user", "content": model_input}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_response = response.choices[0].message.content.strip()

    # Step 4: FORCE Bengali output
    translated_response = translate_en_to_bn(assistant_response)

    # Step 5: TTS cleanup
    cleaned_response = clean_text_for_tts(translated_response)

    audio_fp = BytesIO()
    gTTS(text=cleaned_response, lang="bn", slow=False).write_to_fp(audio_fp)
    audio_fp.seek(0)

    audio_base64 = base64.b64encode(audio_fp.read()).decode()

    return jsonify({
        "response": translated_response,
        "audio": audio_base64,
        "call": None
    })

@app.route("/ping")
def ping():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)