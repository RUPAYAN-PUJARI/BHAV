import ssl
import requests
from flask_cors import CORS
from flask import Flask, request, jsonify
from gtts import gTTS
import base64
import os
from io import BytesIO
from groq import Groq  # type: ignore
from deep_translator import LibreTranslator
import re  # Added for text cleaning

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

def custom_translate(text, to_lang="en", from_lang="bn"):
    """
    Uses LibreTranslate via deep-translator
    Free, open-source, and significantly better than mtranslate for Bengali.
    """
    if not text.strip():
        return text

    try:
        return LibreTranslator(
            source=from_lang,
            target=to_lang
        ).translate(text)
    except Exception as e:
        print("Translation error:", e)
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

    # Normalize input: ALWAYS translate to English
    translated_prompt = custom_translate(user_prompt, "en", "auto")

    # Groq interaction (force English)
    messages = [
        {
            "role": "system",
            "content": (
                "A polite and helpful assistant."
                "You are an assistant that MUST respond ONLY in English. "
                "Do not use any other language."
            )
        },
        {
            "role": "user",
            "content": translated_prompt
        }
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_response = response.choices[0].message.content

    # FORCE output to Bengali (no conditions)
    translated_response = custom_translate(assistant_response, "bn", "en")

    cleaned_response = clean_text_for_tts(translated_response)

    audio_fp = BytesIO()
    tts = gTTS(text=cleaned_response, lang="bn", slow=False, tld="com")
    tts.write_to_fp(audio_fp)
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
