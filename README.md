# Chest X-Ray Pneumonia Detection

ResNet18 fine-tuned on the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) to classify chest X-Rays as **NORMAL** or **PNEUMONIA**.

---

## Project structure

```
chest_xray_pneumonia_detection/
│
├── chest_xray_pneumonia.py   ← training + evaluation script
├── Dockerfile.trainer        ← container for model training
├── docker-compose.yml        ← orchestrate all 3 microservices
│
├── api/
│   └── app.py                ← FastAPI inference server (port 8000)
│
├── ui/
│   ├── streamlit_app.py      ← Streamlit web UI (port 8501)
│   └── Dockerfile
│
├── docker/
│   └── Dockerfile            ← API container
│
├── requirements.txt
└── README.md
```

---

## Microservices architecture

```
[model-trainer] ──saves weights──▶ [/models volume]
                                          │
                                    [inference-api] ◀── POST /predict
                                          ▲
                                        [ui] ◀── browser :8501
```

---

## Quick start

## Docker Compose (recommended)

```bash
# Train + launch all services in one command
docker compose up --build

# Open http://localhost:8501 when everything is healthy.

# If the model is already trained, skip training:
docker compose up inference-api ui
```

| Service | URL | Description |
|---|---|---|
| `inference-api` | http://localhost:8000 | FastAPI — POST /predict |
| `ui` | http://localhost:8501 | Streamlit web interface |
| `model-trainer` | — | Runs once, saves weights to shared volume |


## Key improvements over original

| Area | Original | Improved |
|---|---|---|
| Data split | 90 / 0.2 / 9.8 (only 16 val images) | 85 / 0.85 / 14 (58 val, 862 test) |
| Fine-tuning | FC layer only | **layer4 + FC** (partial fine-tune) |
| Head | Linear → ReLU → Dropout × 2 | Linear → **BN** → ReLU → Dropout × 2 |
| Loss | CrossEntropyLoss | **CrossEntropyLoss + label smoothing 0.1** |
| Scheduler | ReduceLROnPlateau | **CosineAnnealingLR** |
| Inference | Single pass | **Test-Time Augmentation (TTA ×5)** |
| API response | prediction + confidence | + **per-class probabilities** |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | health check |
| POST | `/predict` | upload image → get prediction |

**Example response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9741,
  "probabilities": {
    "NORMAL": 0.0259,
    "PNEUMONIA": 0.9741
  }
}
```

---

## Dataset

- Source: [paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Total images: ~5,856
- Split used: Train 4,936 · Val 58 · Test 862

> Disclaimer: This model is for research and educational purposes only. It is not a certified medical device and should not be used for clinical diagnosis.
