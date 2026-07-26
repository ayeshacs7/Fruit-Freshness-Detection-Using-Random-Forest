import streamlit as st
from PIL import Image
import tempfile
import numpy as np

from feature_extraction import extract_features
from utils.model_utils import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fruit Freshness Detection",
    page_icon="🍎",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = load_model()


CLASS_NAMES = {
    0: "🍎 Fresh Apple",
    1: "🍎 Rotten Apple",
    2: "🍌 Fresh Banana",
    3: "🍌 Rotten Banana",
    4: "🍓 Fresh Strawberry",
    5: "🍓 Rotten Strawberry"
}

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Fruit Freshness Detection")

st.sidebar.markdown("""
### Model Information

**Algorithm:** Random Forest

**Accuracy:** 87.72%
""")

# -----------------------------
# Main Page
# -----------------------------
st.title("🍎 Fruit Freshness Detection")

st.write("Upload a fruit image to predict whether it is **Fresh** or **Rotten**.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)
st.write("Debug 1: App Loaded")

if uploaded_file is not None:
    st.write("Debug 2: File Selected")
    st.write(uploaded_file.name)

if uploaded_file is not None:

    try:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            temp_path = temp.name

        # Extract features
        features = extract_features(temp_path)

        if features is None:
            st.error("Feature extraction failed.")
            st.stop()
            st.write("Debug 3: Extracting Features")

        # Prediction
        prediction = model.predict([features])[0]

        st.success(f"Prediction: {CLASS_NAMES[prediction]}")
        st.write("Debug 4: Prediction Completed")

        # Confidence
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([features])[0]
            confidence = np.max(probabilities) * 100

            st.info(f"Confidence: {confidence:.2f}%")

            st.progress(int(confidence))

    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)