"""
Chest X-Ray Pneumonia Detection — Streamlit UI
Run:  streamlit run ui/streamlit_app.py
"""

import streamlit as st
import requests
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered",
)

st.title("🫁 Chest X-Ray Pneumonia Detection")
st.caption("Upload a chest X-Ray image. The model will predict **NORMAL** or **PNEUMONIA**.")

uploaded_file = st.file_uploader(
    "Choose an X-Ray image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_column_width=True)

    with col2:
        st.write("")
        st.write("")
        if st.button("🔍 Analyze", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                        timeout=30,
                    )
                    result = response.json()

                    label = result["prediction"]
                    conf  = result["confidence"]
                    probs = result.get("probabilities", {})

                    if label == "PNEUMONIA":
                        st.error(f"⚠️  **{label}**")
                    else:
                        st.success(f"✅  **{label}**")

                    st.metric("Confidence", f"{conf:.2%}")

                    if probs:
                        st.write("**Class probabilities:**")
                        st.progress(probs.get("PNEUMONIA", 0))
                        st.caption(
                            f"NORMAL: {probs.get('NORMAL', 0):.2%} │ "
                            f"PNEUMONIA: {probs.get('PNEUMONIA', 0):.2%}"
                        )

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure it's running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {e}")

st.divider()
st.caption(
    "Model: ResNet18 fine-tuned with Test-Time Augmentation (TTA). "
    "For research purposes only — not a medical device."
)
