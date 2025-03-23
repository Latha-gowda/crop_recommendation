from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
import tensorflow as tf
import numpy as np
import joblib
from pydantic import BaseModel

# Load trained model & preprocessors
model = tf.keras.models.load_model("crop_recommendation_nn_final.h5")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

# Initialize FastAPI
app = FastAPI()

# Define API Key for Authentication
API_KEY = "mysecureapikey123"  # Change this to your secure key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Function to verify API Key
def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Define input schema (Without Humidity)
class CropRequest(BaseModel):
    water_level: float
    soil_type: str
    land_area: float
    location: str
    temperature: float
    season: str

# API Endpoint for Crop Prediction (Protected by API Key)
@app.post("/predict")
def predict_crop(data: CropRequest, api_key: str = Depends(get_api_key)):
    # Convert input to array format
    input_data = np.array([
        data.water_level,
        label_encoders["Soil Type"].transform([data.soil_type])[0],
        data.land_area,
        label_encoders["Location"].transform([data.location])[0],
        data.temperature,
        label_encoders["Season"].transform([data.season])[0]
    ]).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)  # Get highest probability class
    recommended_crop = crop_encoder.inverse_transform([predicted_class])[0]  # Convert back to crop name

    return {"Recommended Crop": recommended_crop}

# Run API using: uvicorn api:app --reload
