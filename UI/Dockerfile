FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests pillow

COPY ui/streamlit_app.py .

ENV API_URL=http://inference-api:8000

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
