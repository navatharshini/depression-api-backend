import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, Any

app = FastAPI(
    title="CheerBuddy Depression Prediction API",
    description="API for predicting depression risk based on lifestyle factors",
    version="1.0.1"
)

# Global variables for lazy loading
model = None
scaler = None
pca = None
feature_names = None

def load_model_artifacts():
    """Lazy load model and preprocessing objects"""
    global model, scaler, pca, feature_names
    
    if model is None:
        try:
            with open("depression_model.pkl", "rb") as f:
                data = pickle.load(f)
            model = data["model"]
            scaler = data["scaler"]
            pca = data["pca"]
            feature_names = data["feature_names"]
            
            # Verify all components loaded
            if None in [model, scaler, pca, feature_names]:
                raise ValueError("Missing components in pickle file")
                
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

class InputData(BaseModel):
    Gender: int
    Age: int
    Sleep_Duration: float
    Dietary_Habits: int
    Have_you_ever_had_suicidal_thoughts: int
    Family_History_of_Mental_Illness: int
    Financial_Stress: int
    Working_Professional_or_Student: int
    Weekly_Exercise: int
    Stress_Level: int
    Academic_Pressure: int
    Study_Satisfaction: int
    CGPA: float
    Work_Pressure: int
    Job_Satisfaction: int
    
    class Config:
        schema_extra = {
            "example": {
                "Gender": 1,
                "Age": 25,
                "Sleep_Duration": 7.5,
                "Dietary_Habits": 3,
                "Have_you_ever_had_suicidal_thoughts": 0,
                "Family_History_of_Mental_Illness": 1,
                "Financial_Stress": 2,
                "Working_Professional_or_Student": 1,
                "Weekly_Exercise": 2,
                "Stress_Level": 3,
                "Academic_Pressure": 2,
                "Study_Satisfaction": 4,
                "CGPA": 3.5,
                "Work_Pressure": 2,
                "Job_Satisfaction": 3
            }
        }

@app.on_event("startup")
async def startup_event():
    """Preload model on startup if in production"""
    if os.getenv("ENVIRONMENT") == "production":
        load_model_artifacts()

@app.get("/")
def home():
    return {"message": "Welcome to CheerBuddy Depression Prediction API"}

@app.get("/health")
async def health_check():
    """Endpoint for health checks"""
    try:
        load_model_artifacts()  # Test model loading without prediction
        return {"status": "healthy", "model_loaded": model is not None}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/predict", response_model=Dict[str, Any])
async def predict(data: InputData):
    """
    Predict depression risk based on input parameters
    
    Returns:
        - prediction: 0 (no depression) or 1 (depression)
        - probabilities: Confidence scores for each class
    """
    try:
        # Lazy load model on first request
        load_model_artifacts()
        
        # 1. Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # 2. Add missing features with default values
        for feat in set(feature_names) - set(input_df.columns):
            input_df[feat] = 0
        
        # 3. Ensure correct feature order
        input_df = input_df[feature_names]
        
        # 4. Apply preprocessing
        scaled_features = scaler.transform(input_df)
        pca_features = pca.transform(scaled_features)
        
        # 5. Make prediction
        prediction = model.predict(pca_features)[0]
        probabilities = model.predict_proba(pca_features)[0]
        
        return {
            "prediction": int(prediction),
            "probabilities": {
                "no_depression": float(probabilities[0]),
                "depression": float(probabilities[1])
            },
            "model_version": "1.0.1"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
            headers={"X-Error": "Internal Server Error"}
        )

# Add CORS middleware if needed
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)