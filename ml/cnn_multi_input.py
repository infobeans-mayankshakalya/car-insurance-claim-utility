from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_multi_task_model(meta_input_dim: int):
    # Load MobileNetV2 without top classifier
    base_model = MobileNetV2(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",   # Global average pooling instead of Flatten
        weights="imagenet"  # Use pretrained ImageNet weights
    )

    # Freeze backbone first (optional: unfreeze later for fine-tuning)
    base_model.trainable = False

    # Image branch
    image_input = base_model.input
    x = base_model.output
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Metadata branch
    meta_input = Input(shape=(meta_input_dim,), name="car_metadata")
    m = Dense(64, activation="relu")(meta_input)
    m = Dropout(0.2)(m)

    # Fusion
    combined = Concatenate()([x, m])

    # Severity output (classification)
    severity_output = Dense(3, activation="softmax", name="severity_output")(combined)

    # Cost output (regression)
    cost_output = Dense(1, activation="linear", name="cost_output")(combined)

    # Final model
    model = Model(inputs=[image_input, meta_input], outputs=[severity_output, cost_output])

    model.compile(
        optimizer="adam",
        loss={
            "severity_output": "categorical_crossentropy",
            "cost_output": "mse",
        },
        loss_weights={
            "severity_output": 5.0,   # give more weight to classification
            "cost_output": 0.1,       # scale down regression
        },
        metrics={
            "severity_output": "accuracy",
            "cost_output": "mae",
        }
    )

    return model
