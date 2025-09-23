import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb

# Load CNN classifier
cnn_model = load_model("ml/cnn_damage_classifier.h5")

# Load cost model and encoder
cost_model = joblib.load("ml/cost_model.joblib")
encoder = joblib.load("ml/encoder.joblib")

# Damage classes used in CNN
CNN_CLASSES = ["minor_dent", "scratch", "moderate_dent", "severe_dent"]

def classify_damage(img_bytes: bytes):
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # CNN input size
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img)[0]  # softmax
    class_idx = int(np.argmax(pred))
    confidence = float(pred[class_idx])
    label = CNN_CLASSES[class_idx]

    # Map label to severity
    if "minor" in label or "scratch" in label:
        severity = "minor"
    elif "moderate" in label:
        severity = "moderate"
    else:
        severity = "severe"

    return {"label": label, "confidence": confidence, "severity": severity}

def predict_cost(make: str, model_name: str, year: int, damage_label: str, severity: str):
    X = [[make, model_name, year, damage_label, severity]]
    X_enc = encoder.transform(X)  # Preprocessing: OneHotEncoder, etc.

    # Convert to DMatrix
    dmatrix = xgb.DMatrix(X_enc)
    cost = cost_model.predict(dmatrix)[0]
    return float(cost)
