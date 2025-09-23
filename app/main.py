from fastapi import FastAPI, UploadFile, File, Form
from app.interface import classify_damage, predict_cost
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/inference/local")
async def inference_local(
    file: UploadFile = File(...),
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...)
):
    img_bytes = await file.read()

    # Run CNN classification
    damage_info = classify_damage(img_bytes)
    label = damage_info["label"]
    severity = damage_info["severity"]
    confidence = damage_info["confidence"]

    # Predict cost
    cost_estimate = predict_cost(make, model, year, label, severity)

    return {
        "damage_label": label,
        "confidence": confidence,
        "severity": severity,
        "cost_estimate": cost_estimate,
        "car": {"make": make, "model": model, "year": year}
    }
