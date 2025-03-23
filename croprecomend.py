import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
import joblib

# Load the balanced dataset
df = pd.read_csv("crop_recommendation_balanced.csv")

# Encode categorical features (Soil Type, Location, Season)
label_encoders = {}
for col in ["Soil Type", "Location", "Season"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later

# Extract features (X) and target (y)
X = df.drop(columns=["Crop Label"])  # Features
y = df["Crop Label"]  # Target (single-label after balancing)

# Encode target labels
le_crop = LabelEncoder()
y_encoded = le_crop.fit_transform(y)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 🚀 Build Improved Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    keras.layers.BatchNormalization(),  # Normalize for stable training
    keras.layers.Dropout(0.3),  # Prevent overfitting

    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(len(le_crop.classes_), activation='softmax')  # Output layer (multi-class classification)
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Accuracy
test_loss, test_accuracy = model.evaluate(X_train, y_train)
print(f"Improved Model Accuracy: {test_accuracy:.2f}")

# Save Model & Encoders
model.save("crop_recommendation_nn_final.h5")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

