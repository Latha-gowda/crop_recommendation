from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import joblib
import os
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Define paths for model and preprocessing files
model_path = "crop_recommendation_model.keras"
scaler_path = "scaler.pkl"
label_encoders_path = "label_encoders.pkl"
crop_encoder_path = "crop_encoder.pkl"

# Check if files exist before loading
for file in [model_path, scaler_path, label_encoders_path, crop_encoder_path]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Error: '{file}' not found! Please check the file path.")

# Load trained model & preprocessors
print("Loading model and preprocessing files...")  # Debug log
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(label_encoders_path)
crop_encoder = joblib.load(crop_encoder_path)
print("Model and files loaded successfully!")  # Debug log

# Define input schema
class CropRequest(BaseModel):
    water_level: float
    soil_type: str
    land_area: float
    location: str
    temperature: float
    season: str

# Health Check Endpoint
@app.get("/")
def root():
    return {"message": "FastAPI is working!"}

# API Endpoint for Crop Prediction
@app.post("/predict")
def predict_crop(data: CropRequest):
    try:
        # Ensure label encoders have correct keys
        expected_keys = ["Soil Type", "Location", "Season"]
        for key in expected_keys:
            if key not in label_encoders:
                raise KeyError(f"Missing key '{key}' in label_encoders.pkl")

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run API using: uvicorn api:app --reload --port 8080

