# from fastapi import FastAPI, File, Form, UploadFile
# from keras.saving import load_model
# import numpy as np
# import os, pickle
# from fastapi.middleware.cors import CORSMiddleware
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt

# from ml.preprocess import preprocess_image, encode_metadata

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- Load multi-task model ---
# MODEL_PATH = "ml/cnn_multi_task.keras"
# model = load_model(MODEL_PATH)
# ENCODER_PATH = "ml/meta_encoder.pkl"
# with open(ENCODER_PATH, "rb") as f:
#     meta_encoder = pickle.load(f)

# # Severity labels mapping
# labels = ["minor", "moderate", "severe"]

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...),
#     make: str = Form(...),
#     model_name: str = Form(...),
#     year: int = Form(...),
#     vehicle_type: str = Form(...)
# ):
#     # --- Save uploaded file temporarily ---
#     os.makedirs("tmp", exist_ok=True)
#     image_path = f"tmp/{file.filename}"
#     with open(image_path, "wb") as f:
#         f.write(await file.read())

#     # --- Preprocess ---
#     img = preprocess_image(image_path)
#     img = np.expand_dims(img, axis=0)

#     heatmap, predicted_class = grad_cam(model, img)

#     meta = encode_metadata(make, model_name, year, vehicle_type, fit=False, encoder=meta_encoder)
#     meta = np.expand_dims(meta, axis=0)

#     # --- Inference (two outputs) ---
#     severity_pred, cost_pred = model.predict([img, meta])

#     # Severity
#     label_idx = int(np.argmax(severity_pred, axis=1)[0])
#     confidence = float(np.max(severity_pred))
#     severity = labels[label_idx]

#     # Cost
#     cost_estimate = float(cost_pred[0][0])  # regression output

#     plt.imshow(meta)
#     plt.imshow(heatmap, cmap="jet", alpha=0.5)
#     plt.title(f"Predicted class: {predicted_class}")
#     plt.savefig()

#     return {
#         "damage_label": severity,
#         "confidence": confidence,
#         "severity": severity,
#         "cost_estimate": cost_estimate,
#         "car": {
#             "make": make,
#             "model": model_name,
#             "year": year,
#             "type": vehicle_type
#         }
#     }


# def grad_cam(model, img_array, layer_name="Conv_1", class_idx=None):
#     # Get model layers
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(layer_name).output, model.output[0]]  # severity head
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model([img_array])
#         if class_idx is None:
#             class_idx = np.argmax(predictions[0])
#         loss = predictions[:, class_idx]

#     grads = tape.gradient(loss, conv_outputs)[0]
#     conv_outputs = conv_outputs[0]
#     weights = tf.reduce_mean(grads, axis=(0, 1))

#     cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
#     for i, w in enumerate(weights):
#         cam += w * conv_outputs[:, :, i]

#     cam = np.maximum(cam, 0)
#     cam = cv2.resize(cam.numpy(), (224, 224))
#     cam = cam / cam.max()

#     return cam, class_idx


import io
import base64
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, Form, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
from ml.preprocess import preprocess_image, encode_metadata
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + encoder once
model = load_model("ml/cnn_multi_task.keras")
meta_encoder = joblib.load("ml/meta_encoder.pkl")

def grad_cam(model, img_array, meta, layer_name="Conv_1", class_idx=None):
    """Generate Grad-CAM heatmap for severity prediction."""
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, meta ])
        if class_idx is None:
            class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = cam / cam.max()
    return cam

def overlay_heatmap(img_array, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_array.astype("uint8"), 0.6, heatmap, 0.4, 0)
    return superimposed_img

@app.post("/predict")
async def predict(
    file: UploadFile,
    make: str = Form(...),
    model_name: str = Form(...),
    year: int = Form(...),
    vehicle_type: str = Form(...)
):
    try:
        # ---- Preprocess image ----
        file_bytes = await file.read()
        img, orig_img = preprocess_image(file_bytes, return_original=True)  # orig for overlay

        # ---- Preprocess metadata ----
        meta = encode_metadata(make, model_name, year, vehicle_type, fit=False, encoder=meta_encoder)
        meta = np.expand_dims(meta, axis=0)

        # ---- Predict ----
        severity_probs, cost_pred = model.predict([img, meta])
        severity_idx = int(np.argmax(severity_probs[0]))
        severity_labels = ["minor", "moderate", "severe"]
        severity = severity_labels[severity_idx]
        confidence = float(np.max(severity_probs[0]))
        cost_estimate = float(cost_pred[0][0])

        # ---- Grad-CAM ----
        heatmap = grad_cam(model, img, meta, class_idx=severity_idx)
        highlighted = overlay_heatmap(orig_img, heatmap)

        # Encode highlighted image as base64 for JSON transport
        _, buffer = cv2.imencode(".jpg", highlighted)
        heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

        # return {
        #     "damage_label": severity,
        #     "confidence": confidence,
        #     "severity": severity,
        #     "cost_estimate": cost_estimate,
        #     "car": {
        #         "make": make,
        #         "model": model_name,
        #         "year": year,
        #         "type": vehicle_type
        #     },
        #     "heatmap": heatmap_b64
        # }
    
        return {
            "damage_label": severity,
            "confidence": confidence,
            "severity": severity,
            "cost_estimate": cost_estimate,
            "car": {
                "make": make,
                "model": model_name,
                "year": year,
                "type": vehicle_type
            },
            "heatmap": heatmap_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
