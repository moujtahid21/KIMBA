import logging
import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import traceback
from kimba_ensemble import KIMBAEnsemble
import download_models

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Download models ---
@st.cache_resource
def ensure_models_downloaded():
    try:
        # Appel du script Python pour télécharger les modèles
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "download_models.py"])
        logger.info("✅ Models downloaded successfully!")
    except Exception as e:
        logger.error("Failed to download models. Here is the full traceback:")
        logger.error(traceback.format_exc())
        st.error(f"Failed to download models: {e}")

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# --- Model Loader ---
@st.cache_resource
def load_kimba_model():
    try:
        model = KIMBAEnsemble()
        # The file path is hardcoded. Make sure it is correct.
        model_dir_path = "/workspaces/KIMBA/saved_models"
        
        # This is where the error was happening
        if not os.path.exists(model_dir_path):
            logger.error(f"Directory not found: {model_dir_path}")
            st.error(f"Directory not found: {model_dir_path}")
            return None
        
        model.load_saved_models(model_dir_path)
        model.to(device)
        model.eval()
        logger.info("✅ KIMBA ensemble loaded successfully!")
        return model
    except Exception as e:
        logger.error("Failed to load KIMBA ensemble. Here is the full traceback:")
        logger.error(traceback.format_exc())
        st.error(f"Failed to load KIMBA ensemble: {e}")
        return None

kimba_model = load_kimba_model()

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image, size=(224, 224)) -> np.ndarray:
    image = image.convert("RGB").resize(size)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # (C, H, W) for PyTorch
    return arr

# --- Prediction ---
def predict_image(model, image_arr: np.ndarray):
    with torch.no_grad():
        tensor = torch.tensor(image_arr).unsqueeze(0).to(device)  # (1, C, H, W)
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        label = model.idx_to_label(pred_idx) if hasattr(model, "idx_to_label") else str(pred_idx)
        confidence = float(probs[pred_idx])
        return label, confidence, probs

# --- Streamlit UI ---
st.set_page_config(
    page_title="KIMBA Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 KIMBA Tumor Detection App")
st.markdown("Upload a tumor scan image. The KIMBA ensemble will predict the tumor type and confidence.")

with st.sidebar:
    st.header("About KIMBA")
    st.info(
        "KIMBA is an ensemble model based on two models trained on different datasets. "
        "It enables robust detection of tumor types from medical images."
    )
    st.markdown("---")
    st.subheader("Model Status")
    if kimba_model:
        st.success(f"Model loaded on: {device}")
    else:
        st.error("Failed to load model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_arr = preprocess_image(image)
        if kimba_model:
            label, confidence, probs = predict_image(kimba_model, image_arr)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence for predicted class: {confidence:.2f}")
            st.markdown("---")
            if hasattr(kimba_model, "class_names") and kimba_model.class_names:
                class_scores = {name: float(probs[i]) for i, name in enumerate(kimba_model.class_names)}
            else:
                class_scores = {str(i): float(p) for i, p in enumerate(probs)}
            st.json(class_scores)
        else:
            st.error("Model is not available.")
    except Exception as e:
        logger.error("Error during prediction:")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by Abdelbasset Moujtahid \& Othmane Elmekaoui")
st.markdown("Source code: [GitHub](https://github.com/moujtahid21/KIMBA/tree/master)")
# --- End of Streamlit App ---
