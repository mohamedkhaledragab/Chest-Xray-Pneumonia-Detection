FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
    fastapi "uvicorn[standard]" python-multipart pillow scikit-learn

COPY api/app.py .

ENV MODEL_PATH=/app/models/xray_classifier.pth

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
