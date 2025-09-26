import numpy as np
import cv2, os
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import io
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Global encoders so training and inference share them
make_encoder = LabelEncoder()
model_encoder = LabelEncoder()
type_encoder = LabelEncoder()

# def preprocess_image(image_path, target_size=(224, 224)):
#     """Load and preprocess an image."""
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, target_size)
#     img = img.astype("float32") / 255.0  # normalize
#     return img

def preprocess_image(file_bytes, return_original=False):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    proc_array = preprocess_input(img_array)
    # proc_array = np.expand_dims(proc_array, axis=0)
    
    if return_original:
        return proc_array, img_array.astype("uint8")
    return proc_array

# def encode_metadata(make, model, type, year, fit=False):
#     """
#     Encode car metadata into numeric vector.
#     Args:
#         make (str): car make
#         model (str): car model
#         type (str): car type
#         year (int): car year
#         fit (bool): if True, fit encoders (only during training).
#     """
#     global make_encoder, model_encoder, type_encoder

#     if fit:
#         make_encoder.fit([make])
#         model_encoder.fit([model])
#         type_encoder.fit([type])

#     make_encoded = make_encoder.transform([make])[0] if make in make_encoder.classes_ else 0
#     model_encoded = model_encoder.transform([model])[0] if model in model_encoder.classes_ else 0
#     type_encoded = type_encoder.transform([type])[0] if type in type_encoder.classes_ else 0
#     year_norm = (year - 2000) / 25.0  # normalize years ~2000–2025

#     return np.array([make_encoded, model_encoded, type_encoded, year_norm])

# --- Metadata encoder ---
# Save encoders so inference matches training
ENCODER_PATH = "ml/meta_encoder.pkl"

class MetadataEncoder:
    def __init__(self):
        self.encoder = None

    def fit(self, X):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(X)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(self.encoder, f)

    def load(self):
        with open(ENCODER_PATH, "rb") as f:
            self.encoder = pickle.load(f)

    def transform(self, X):
        return self.encoder.transform(X)

meta_encoder = MetadataEncoder()

def encode_metadata(make, model, year, vehicle_type, fit=False, encoder=None):
    X = np.array([[make, model, year, vehicle_type]])

    if encoder is not None:
        return encoder.transform(X)[0]

    if fit:
        meta_encoder.fit(X)
    else:
        if meta_encoder.encoder is None:
            meta_encoder.load()
    return meta_encoder.transform(X)[0]