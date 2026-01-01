CODE_DIR = "sagemaker_prophet_inference"
os.makedirs(CODE_DIR, exist_ok=True)

inference_py = r'''
import os
import json
import pickle
import pandas as pd

MODEL_FILENAME = "model.pkl"

def model_fn(model_dir):
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    # Expect JSON: {"ds": ["2023-12-01", "2024-01-01", ...]}
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        ds = payload.get("ds", [])
        if not isinstance(ds, list):
            raise ValueError('"ds" must be a list of date strings.')
        return pd.DataFrame({"ds": pd.to_datetime(ds)})
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    pred = model.predict(input_data)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        out = prediction.copy()
        out["ds"] = out["ds"].dt.strftime("%Y-%m-%d")
        return out.to_json(orient="records"), response_content_type
    raise ValueError(f"Unsupported accept type: {response_content_type}")
'''
with open(os.path.join(CODE_DIR, "inference.py"), "w") as f:
    f.write(inference_py)

req_txt = """prophet
cmdstanpy
pandas
numpy
"""
with open(os.path.join(CODE_DIR, "requirements.txt"), "w") as f:
    f.write(req_txt)

print("Wrote:", os.path.join(CODE_DIR, "inference.py"))
print("Wrote:", os.path.join(CODE_DIR, "requirements.txt"))
