# Import required libraries.
import json
import joblib
import os
from datetime import datetime
# Directory where the trained model and metadata will be stored.
MODEL_DIR = "model"

def save_model(model, filename="fruit_freshness_model.pkl"):

# Create the model directory if it does not already exist.
    os.makedirs(MODEL_DIR, exist_ok=True)
#Create the complete file path for saving the trained model.
    path = os.path.join(MODEL_DIR, filename)
# Save the trained machine learning
# model using Joblib.
    joblib.dump(model, path)
# Display a confirmation message.
    print(f"Model saved: {path}")


def save_metadata(accuracy, algorithm):
 # Store important information about the trained model.
    metadata = {
        "algorithm": algorithm,
        "accuracy": round(accuracy * 100, 2),
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
# Define the metadata file path.
    path = os.path.join(MODEL_DIR, "metadata.json")
# Save metadata as a JSON file.
    with open(path, "w") as file:
        json.dump(metadata, file, indent=4)
# Display a confirmation message.
    print("Metadata saved.")

def load_model(filename="fruit_freshness_model.pkl"):
# Create the complete file path of the trained model.
    path = os.path.join(MODEL_DIR, filename)
# Load the trained machine learning model.
    model = joblib.load(path)
# Return the loaded model for prediction.
    return model