import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the model and preprocessing objects
with open("depression_model.pkl", "rb") as f:
    data = pickle.load(f)
    
model = data["model"]
scaler = data["scaler"]
pca = data["pca"]
feature_names = data["feature_names"]  # This should contain all 73 feature names

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

@app.post("/predict")
async def predict(data: InputData):
    try:
        # 1. Convert input to DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # 2. Replicate ALL feature engineering from training
        # This must exactly match what you did during training
        # Example for one-hot encoding:
        # input_df = pd.get_dummies(input_df, columns=['categorical_var'])
        
        # 3. Ensure you have exactly 84 features at this point
        # Add missing features with default values if needed
        missing_features = set(feature_names) - set(input_df.columns)
        for feat in missing_features:
            input_df[feat] = 0  # or appropriate default
        
        # 4. Reorder features to match training order
        input_df = input_df[feature_names]
        
        # 5. Apply preprocessing in correct order
        scaled_features = scaler.transform(input_df)  # Now has 84 features
        pca_features = pca.transform(scaled_features)  # Reduces to 73 features
        prediction = model.predict(pca_features)[0]
      
        probabilities = model.predict_proba(pca_features)[0]
        
        return {"prediction": int(prediction),
"probabilities": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            }
                
                }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))