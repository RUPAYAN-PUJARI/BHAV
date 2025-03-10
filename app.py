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
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from mtranslate import translate  # type: ignore
import re  # For language detection and text cleaning

# Download VADER lexicon if not already available
nltk.download('vader_lexicon')

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

# Load API key from config file
with open("config.json") as config_file:
    config_data = json.load(config_file)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def custom_translate(text, to_lang="en", from_lang="bn"):
    """ Translates text between Bengali and English. """
    return translate(text, to_lang, from_lang)

def clean_text_for_tts(text):
    """ Cleans the AI response by removing unnecessary symbols before speech synthesis. """
    text = re.sub(r'[*=\\/(){}<>]', '', text)  # Remove unwanted characters
    return re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

def is_bengali(text):
    """ Checks if the text contains Bengali characters. """
    return bool(re.search(r'[\u0980-\u09FF]', text))

# Endpoint to handle chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("prompt", "")

    # Detect input language
    bengali_input = is_bengali(user_prompt)

    # Translate to English if the input is in Bengali
    translated_prompt = custom_translate(user_prompt, 'en', 'bn') if bengali_input else user_prompt

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(translated_prompt)
    sentiment_compound = sentiment_scores["compound"]
    is_negative = sentiment_compound < -0.5  # Threshold for negativity

    # Debugging: Print sentiment scores
    print(f"Sentiment Analysis Scores: {sentiment_scores}")

    # Interact with the LLM model
    messages = [
        {"role": "system", "content": "A helpful polite assistant."},
        {"role": "user", "content": translated_prompt}
    ]
    response = client.chat.completions.create(model="llama-3.2-3b-preview", messages=messages)
    assistant_response = response.choices[0].message.content

    # Translate back to Bengali if the input was in Bengali
    final_response = custom_translate(assistant_response, 'bn', 'en') if bengali_input else assistant_response

    # Modify response if sentiment is negative
    call_number = None
    if is_negative:
        final_response += "\n\n(সতর্কতা জরুরি নম্বর)" if bengali_input else "\n\n(WARNING: Emergency number)"
        call_number = "1098"

    # Clean response before speech synthesis
    cleaned_response = clean_text_for_tts(final_response)

    # Set TTS language based on input
    tts_lang = 'bn' if bengali_input else 'en'

    # Convert text to speech
    audio_fp = BytesIO()
    tts = gTTS(text=cleaned_response, lang=tts_lang, slow=False, tld="com")
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)

    # Encode audio to base64
    audio_base64 = base64.b64encode(audio_fp.read()).decode()

    # Return response with sentiment scores
    return jsonify({
        "response": final_response,
        "audio": audio_base64,
        "call": call_number,
        "sentiment": {
            "compound": sentiment_scores["compound"],
            "positive": sentiment_scores["pos"],
            "negative": sentiment_scores["neg"],
            "neutral": sentiment_scores["neu"]
        }
    })

if __name__ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)