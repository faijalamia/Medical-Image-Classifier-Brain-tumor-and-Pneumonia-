import streamlit as st
import numpy as np
import base64
from PIL import Image
from Models import classifier  # Your TensorFlow-based classifier


# --- Classification Model Data (from Telegram bot version) ---
data = {
    "Brain Tumor": {
        "class_names": ['NO BRAIN TUMOR', 'BRAIN TUMOR'],
        "weights_name": "brain_tumor.keras",
    },
    "Chest X-Ray": {
        "class_names": ["NORMAL", "PNEUMONIA"],
        "weights_name": "chest_xray.keras",
    }
}

# --- Page Configuration ---
st.set_page_config(page_title="Medical Image Classifier", layout="wide")


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

bg_image_path = "img1.png"  # Adjust path as needed
bg_image_encoded = get_base64_image(bg_image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:jpg;base64,{bg_image_encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
"""
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #242121;
        font-weight: bold;
        margin-top: -60px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="title">🧠 Medical Image Classification</div>', unsafe_allow_html=True)

# --- Model Selection ---
model_name = st.selectbox("🔍 Choose a Classification Model", list(data.keys()))
model_data = data[model_name]

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    prediction = None

    col1, col2, col3 = st.columns([1.2, 1, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="📥 Uploaded Image")

    with col2:
        st.markdown("""<br><br>""", unsafe_allow_html=True)

        st.markdown("""
            <style>
            div.stButton > button {
                margin-left: 30%;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("🚀 Run Classification"):
            with st.spinner("Classifying..."):
                label, score = classifier(img, model_data["weights_name"], model_data["class_names"])
                prediction = (label, score)

    with col3:
        if prediction:
            label, score = prediction
            st.markdown(f"""
                <br><br>
                <h3>📷 Prediction: <code>{label}</code></h3>
                <h4>🔢 Confidence: <code>{score:.2f}</code></h4>
                <h5>🧠 Model Used: <code>{model_data['weights_name']}</code></h5>
            """, unsafe_allow_html=True)
