
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import librosa
import numpy as np
import speech_recognition as sr
import spacy
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import uvicorn

# Load the NLP model for similarity checks
nlp = spacy.load("en_core_web_sm")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is working!"}

class RiskScoreResponse(BaseModel):
    risk_score: float

# Helper function to get transcription
def get_transcription(audio_bytes: bytes):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(BytesIO(audio_bytes))
    with audio as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

# Helper function to get pauses (simplified example)
def get_pauses(audio_bytes: bytes):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    intervals = librosa.effects.split(y, top_db=20)
    return len(intervals)

# Helper function to get pitch variability
def get_pitch_variability(audio_bytes: bytes):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
    pitch, _ = librosa.core.piptrack(y=y, sr=sr)
    pitch_variability = np.std(pitch)
    return pitch_variability

# Endpoint for calculating the risk score based on the audio file
@app.post("/calculate_risk_score", response_model=RiskScoreResponse)
async def calculate_risk_score(file: UploadFile = File(...)):
    # Read the audio file
    audio_bytes = await file.read()

    # Extract features
    transcription = get_transcription(audio_bytes)
    pauses = get_pauses(audio_bytes)
    pitch_variability = get_pitch_variability(audio_bytes)

    # Example: Calculate a simple risk score based on features
    # Just for illustration; you can use your own logic
    risk_score = np.random.random()  # Dummy value for illustration

    # Return risk score
    return RiskScoreResponse(risk_score=risk_score)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
