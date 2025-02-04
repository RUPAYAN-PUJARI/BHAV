import ssl
import warnings
import requests
import json
import os
import base64
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from gtts import gTTS
from io import BytesIO
from groq import Groq
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from mtranslate import translate

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "supersecretkey"  # Change to a secure key

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Load API Key
with open("config.json") as config_file:
    config_data = json.load(config_file)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


### DATABASE MODELS ###
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)


### UTILITY FUNCTIONS ###
def custom_translate(text, to_lang="en", from_lang="bn"):
    return translate(text, to_lang, from_lang)


def clean_text_for_tts(text):
    """Removes unnecessary symbols for better TTS output."""
    text = text.replace('*', '')
    text = re.sub(r'[=\\/]', '', text)
    text = re.sub(r'[(){}<>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


### AUTHENTICATION ROUTES ###
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Invalid credentials"}), 401

    access_token = create_access_token(identity=user.id)
    return jsonify({"token": access_token, "message": "Login successful"}), 200


### CHAT ENDPOINT ###
@app.route("/chat", methods=["POST"])
@jwt_required()
def chat():
    user_id = get_jwt_identity()
    data = request.json
    user_prompt = data.get("prompt", "")

    # Translate user's input
    translated_prompt = custom_translate(user_prompt, 'en', 'bn')

    # Sentiment Analysis
    sentiment_score = sia.polarity_scores(translated_prompt)["compound"]
    is_negative = sentiment_score < -0.5

    # Get AI response
    messages = [
        {"role": "system", "content": "A helpful polite assistant."},
        {"role": "user", "content": translated_prompt}
    ]
    response = client.chat.completions.create(model="llama-3.2-3b-preview", messages=messages)
    assistant_response = response.choices[0].message.content

    # Translate response to Bengali
    translated_response = custom_translate(assistant_response, 'bn', 'en')

    # Handle emergency cases
    call_number = None
    if is_negative:
        translated_response += "\n\n(সতর্কতা জরুরি নম্বর)"
        call_number = "1098"

    # Clean and convert response to speech
    cleaned_response = clean_text_for_tts(translated_response)
    audio_fp = BytesIO()
    tts = gTTS(text=cleaned_response, lang='bn', slow=False, tld="com")
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    audio_base64 = base64.b64encode(audio_fp.read()).decode()

    # Store chat history in database
    new_chat = ChatHistory(user_id=user_id, prompt=user_prompt, response=translated_response)
    db.session.add(new_chat)
    db.session.commit()

    return jsonify({"response": translated_response, "audio": audio_base64, "call": call_number})


### CHAT HISTORY ENDPOINT ###
@app.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    history = ChatHistory.query.filter_by(user_id=user_id).all()
    return jsonify([{"prompt": chat.prompt, "response": chat.response} for chat in history])


### RUN SERVER ###
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensure database tables are created
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)