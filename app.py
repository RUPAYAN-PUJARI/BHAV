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

LIBRE_URL = "https://libretranslate.com/translate"

def translate_text(text, source, target):
    if not text.strip():
        return text

    payload = {
        "q": text,
        "source": source,
        "target": target,
        "format": "text"
    }

    try:
        r = requests.post(LIBRE_URL, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()["translatedText"]
    except Exception as e:
        print("Translation failed:", e)
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

    # If Bengali → English
    model_input = translate_text(user_prompt, "bn", "en")

    messages = [
        {
            "role": "system",
            "content": (
                "A polite helpful assistant. "
                "Your final answer MUST be in Bengali only."
            )
        },
        {"role": "user", "content": model_input}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_response = response.choices[0].message.content.strip()

    # ALWAYS English → Bengali
    final_response = translate_text(assistant_response, "en", "bn")

    cleaned = clean_text_for_tts(final_response)

    audio_fp = BytesIO()
    gTTS(text=cleaned, lang="bn").write_to_fp(audio_fp)
    audio_fp.seek(0)

    return jsonify({
        "response": final_response,
        "audio": base64.b64encode(audio_fp.read()).decode()
    })

@app.route("/ping")
def ping():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)