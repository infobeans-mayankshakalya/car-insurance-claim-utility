import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from keras.saving import save_model
from sklearn.model_selection import train_test_split
from cnn_multi_input import build_multi_task_model
from preprocess import preprocess_image, encode_metadata
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Dummy paths (replace with real dataset loader)
IMAGE_DIR = "ml/data/"
LABELS_FILE = "ml/data/car_damage_with_cost.csv"  # assume columns: filename, make, model, year, type, severity, cost

import pandas as pd
df = pd.read_csv(LABELS_FILE)

# --- Prepare datasets ---
images = []
metadata = []
severities = []
costs = []

for _, row in df.iterrows():
    print(os.path.join(IMAGE_DIR, row["severity"], row["image"]))
    with open(os.path.join(IMAGE_DIR, row["severity"], row["image"]), "rb") as f:
        image_bytes = f.read()
        img = preprocess_image(image_bytes)
        images.append(img)

        meta = encode_metadata(row["make"], row["model"], row["year"], row["type"], fit=True)
        metadata.append(meta)

        severities.append(row["severity"])   # e.g. "minor", "moderate", "severe"
        costs.append(row["cost"])            # numeric

images = np.array(images)
metadata = np.array(metadata)
costs = np.array(costs, dtype="float32")

# Encode severity labels
severity_map = {"minor": 0, "moderate": 1, "severe": 2}
y_severity = np.array([severity_map[s] for s in severities])
y_severity = to_categorical(y_severity, num_classes=3)

# Train/test split
X_img_train, X_img_val, X_meta_train, X_meta_val, y_sev_train, y_sev_val, y_cost_train, y_cost_val = train_test_split(
    images, metadata, y_severity, costs, test_size=0.2, random_state=42
)

# --- Build and train ---
# model = build_multi_task_model(meta_input_dim=metadata.shape[1])
model = build_multi_task_model(meta_input_dim=metadata.shape[1])

# history = model.fit(
#     [X_img_train, X_meta_train],
#     {"severity_output": y_sev_train, "cost_output": y_cost_train},
#     validation_data=(
#         [X_img_val, X_meta_val],
#         {"severity_output": y_sev_val, "cost_output": y_cost_val}
#     ),
#     epochs=20,
#     batch_size=32
# )

history = model.fit(
    [X_img_train, X_meta_train],     # list of inputs
    {"severity_output": y_severity, "cost_output": y_cost_train},  # dict of outputs
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    # class_weight={"severity_output": class_weights},  # optional if imbalanced
    callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
)

# Save trained model
os.makedirs("ml", exist_ok=True)
save_model(model, 'ml/cnn_multi_task.keras')

print("✅ Multi-task model trained and saved as cnn_multi_task.keras")
