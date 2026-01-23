import os
import joblib
import numpy as np
import tensorflow as tf
import json

def model_fn(model_dir):
    """Load the model and scaler from the tarball directory"""
    model = tf.keras.models.load_model(os.path.join(model_dir, "cpi_lstm_model.keras"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return {"model": model, "scaler": scaler}

def input_fn(request_body, request_content_type):
    """Parse input data..."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        # Reshape to (1, LOOKBACK, 1) for LSTM
        return np.array(data).reshape(1, -1, 1).astype('float32')
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make prediction and inverse scale back to real CPI units"""
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    
    prediction = model.predict(input_data)
    # Convert back to real dollars
    unscaled_prediction = scaler.inverse_transform(prediction)
    return unscaled_prediction.flatten().tolist()