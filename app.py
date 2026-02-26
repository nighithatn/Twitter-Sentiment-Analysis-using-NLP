# =========================
# FastAPI Sentiment API
# =========================

import re
import pickle
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Constants
# -------------------------
MODEL_PATH = "twitterSentimentModel.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 100

# -------------------------
# Load Model & Tokenizer
# -------------------------
sentimentModel = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="BiLSTM-based sentiment classifier",
    version="1.0"
)

# -------------------------
# Request Schema
# -------------------------
class SentimentRequest(BaseModel):
    text: str

# -------------------------
# Text Cleaning Function
# -------------------------
def cleanText(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# -------------------------
# Health Check
# -------------------------
@app.get("/")
def healthCheck():
    return {"status": "API is running"}

# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predictSentiment(request: SentimentRequest):

    # Clean input
    cleanedText = cleanText(request.text)

    # Tokenize & pad
    sequence = tokenizer.texts_to_sequences([cleanedText])
    paddedSequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post"
    )

    # Predict
    probabilities = sentimentModel.predict(paddedSequence)
    predictedClass = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(np.max(probabilities))

    # Label mapping
    sentimentMapping = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return {
        "inputText": request.text,
        "cleanedText": cleanedText,
        "predictedLabel": predictedClass,
        "sentiment": sentimentMapping[predictedClass],
        "confidence": confidence
    }
