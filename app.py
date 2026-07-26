# Import required libraries.
import streamlit as st
from PIL import Image
import tempfile
import numpy as np
# Import feature extraction and trained model loader.
from feature_extraction import extract_features
from utils.model_utils import load_model
# Configure the Streamlit page title, icon and layout.
st.set_page_config(
    page_title="Fruit Freshness Detection",
    page_icon="🍎",
    layout="centered"
)
# Display the application title and user instructions.
st.title("🍎 Fruit Freshness Detection")
st.write("Upload an image of Apple, Banana or Strawberry.")

# Load the trained machine learning model.
# Stop the application if loading fails.
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model:\n{e}")
    st.stop()
# Map predicted numeric label  to human-readable class names.
CLASS_NAMES = {
    0: "🍎 Fresh Apple",
    1: "🍎 Rotten Apple",
    2: "🍌 Fresh Banana",
    3: "🍌 Rotten Banana",
    4: "🍓 Fresh Strawberry",
    5: "🍓 Rotten Strawberry",
}
# Allow the user to upload a fruit image.
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)
# Continue only if the user uploads an image.
if uploaded_file is not None:
# Open the uploaded image and convert it to RGB format.
    image = Image.open(uploaded_file).convert("RGB")
 # Display the uploaded image.
    st.image(image, caption="Uploaded Image", use_container_width=True)
# Start prediction when the user clicks the button.
    if st.button("Predict"):

        try:

 # Save the uploaded image as a temporary file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                image.save(temp.name)
                temp_path = temp.name

# Extract handcrafted features from the uploaded image.
            features = extract_features(temp_path)
# Stop the prediction proces  if feature extraction fails.
            if features is None:
                st.error("Feature extraction failed.")
                st.stop()
# Convert extracted features into the format expected by the model.
            features = np.array(features).reshape(1, -1)

# Predict the freshness class of the uploaded fruit image.
            prediction = model.predict(features)[0]
# Convert the predicted label into a readable class name.
            result = CLASS_NAMES.get(prediction, "Unknown")
# Display the prediction result.
            st.success(f"Prediction: **{result}**")
 # Display prediction confidence if the model supports probabilities.
            if hasattr(model, "predict_proba"):

                probs = model.predict_proba(features)[0]

                st.subheader("Confidence")
# Display confidence for # every possible class.
                for i, p in enumerate(probs):
                    st.write(f"{CLASS_NAMES[i]} : {p*100:.2f}%")
# Display detailed error information if an unexpected exception occurs.
        except Exception as e:
            st.exception(e)