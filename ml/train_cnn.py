import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Paths
train_dir = "data/train"
val_dir = "data/validation"

# Image generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical"
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(4, activation="softmax")  # 4 classes
])

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save model
os.makedirs("ml", exist_ok=True)
model.save("ml/cnn_damage_classifier.h5")
print("✅ CNN model saved at ml/cnn_damage_classifier.h5")
