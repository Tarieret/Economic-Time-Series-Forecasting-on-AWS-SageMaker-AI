%%writefile inference.py
import os
import joblib
import json
import xgboost as xgb
import numpy as np
import pandas as pd

def model_fn(model_dir):
    """Load model from model_dir"""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "model.json"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))
    return {"model": model, "features": features}

def input_fn(request_body, request_content_type):
    """Parse JSON input"""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """XGBoost predict + reconstruct trend"""
    model = model_dict["model"]
    feature_names = model_dict["features"]
    
    # Convert list of lists to DMatrix
    data = pd.DataFrame(input_data["features"], columns=feature_names)
    dtest = xgb.DMatrix(data)
    
    # Predict change and add to last known value
    predicted_diff = model.predict(dtest)
    last_actual = input_data["last_value"]
    
    # Calculate absolute values
    predictions = last_actual + np.cumsum(predicted_diff)
    return predictions.tolist()

def output_fn(prediction, content_type):
    return json.dumps({"predictions": prediction})