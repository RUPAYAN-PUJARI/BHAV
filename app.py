import ssl
import requests
from flask_cors import CORS
from flask import Flask, request, jsonify
import base64
import os
from io import BytesIO
from groq import Groq  # type: ignore
from mtranslate import translate  # type: ignore
import re
import asyncio
import edge_tts

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")

print("DEBUG: Loaded GROQ_API_KEY from environment:", GROQ_API_KEY[:5] + "***")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def custom_translate(text, to_lang="en", from_lang="bn"):
    return translate(text, to_lang, from_lang)


def clean_text_for_tts(text):
    text = text.replace('*', '')
    text = re.sub(r'[=\\/]', '', text)
    text = re.sub(r'[(){}<>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Async Edge TTS generator
async def generate_speech(text):
    voice = "bn-BD-NabanitaNeural"  # High quality Bengali voice
    communicate = edge_tts.Communicate(text=text, voice=voice)

    audio_stream = BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])

    audio_stream.seek(0)
    return audio_stream


# Endpoint to handle chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("prompt", "")

    # Translate Bengali → English
    translated_prompt = custom_translate(user_prompt, 'en', 'bn')

    messages = [
        {"role": "system", "content": "A helpful polite assistant. Your name is BHAV. When asked about your name, you should say that it is 'BHAV'."},
        {"role": "user", "content": translated_prompt}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_response = response.choices[0].message.content

    # Translate English → Bengali
    translated_response = custom_translate(assistant_response, 'bn', 'en')

    cleaned_response = clean_text_for_tts(translated_response)

    try:
        # Run async TTS inside Flask
        audio_fp = asyncio.run(generate_speech(cleaned_response))
        audio_base64 = base64.b64encode(audio_fp.read()).decode()
    except Exception as e:
        print("TTS Error:", str(e))
        audio_base64 = None

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
