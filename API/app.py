"""
Pneumonia Detection API
========================
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io

# ── Config ──────────────────────────────────────────────────────────
import os

CLASSES    = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = os.getenv("MODEL_PATH", "xray_classifier.pth")
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
TTA_STEPS  = 5   # set to 1 to disable TTA (faster, slightly lower accuracy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build model (must match training architecture) ───────────────────
def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    return model

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ── Transforms ──────────────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

tta_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 10, IMG_SIZE + 10)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── FastAPI ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Upload a chest X-Ray image and get NORMAL / PNEUMONIA prediction.",
    version="2.0",
)


@app.get("/")
def home():
    return {"status": "running", "model": "ResNet18-finetuned", "tta_steps": TTA_STEPS}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG / PNG images accepted.")

    raw = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    # TTA: average TTA_STEPS passes
    prob_sum = 0.0
    with torch.no_grad():
        # First pass: deterministic
        t = base_transform(pil_img).unsqueeze(0).to(device)
        prob_sum += torch.softmax(model(t), dim=1)[0, 1].item()

        # Remaining passes: augmented
        for _ in range(TTA_STEPS - 1):
            t = tta_transform(pil_img).unsqueeze(0).to(device)
            prob_sum += torch.softmax(model(t), dim=1)[0, 1].item()

    pneumonia_prob = prob_sum / TTA_STEPS
    normal_prob    = 1.0 - pneumonia_prob
    pred_idx       = int(pneumonia_prob >= 0.5)

    return JSONResponse({
        "prediction":    CLASSES[pred_idx],
        "confidence":    round(max(pneumonia_prob, normal_prob), 4),
        "probabilities": {
            "NORMAL":    round(normal_prob, 4),
            "PNEUMONIA": round(pneumonia_prob, 4),
        },
    })
