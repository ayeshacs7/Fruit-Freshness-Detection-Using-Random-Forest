# ==========================================================
# Project : Fruit Freshness Detection using Random Forest
# Author  : Ayesha Shafiq
# File    : train_model.py
# Purpose :
# This module loads the dataset, trains the Random Forest
# classifier, evaluates its performance, and saves the
# trained model along with its metadata.
# ==========================================================
# Import required libraries.
import os
import joblib
# Import the Random Forest machine learning algorithm.
from sklearn.ensemble import RandomForestClassifier
# Import evaluation metrics.
from sklearn.metrics import accuracy_score, classification_report
# Import function to split the dataset.
from sklearn.model_selection import train_test_split
# Import custom dataset loader and model utilities.
from utils.dataset_loader import load_dataset
from utils.model_utils import save_model, save_metadata
# ============================
# Configuration
# ============================
# Define the percentage of data
# reserved for model testing.
TEST_SIZE = 0.20
# Set a fixed random stat to
# ensure reproducible results.
RANDOM_STATE = 42
# Path where the trained model
# will be stored.
MODEL_PATH = "model/fruit_freshness_model.pkl"
# ============================
# Load Dataset
# ============================
X, y = load_dataset()
# Display dataset information.
print("\nDataset loaded successfully.")
print(f"Total Samples: {len(X)}")
# ============================
# Split Dataset
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
# Display a message indicating the start of model training.
print("\nTraining Random Forest Model...")
# Create a Random Forest classifier.
# Multiple decision trees are combined to improve prediction accuracy.
model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
# Train the model using the extracted image features.
model.fit(X_train, y_train)
# Generate predictions for the testing dataset.
predictions = model.predict(X_test)
# Calculate the overall classification accuracy.
accuracy = accuracy_score(y_test, predictions)
# Display the model accuracy.
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
# Display detailed performance metrics
# including precision, recall and F1-score.
print(classification_report(y_test, predictions))
# Save the trained machine learning model for future predictions.
save_model(model)
# Save additional model information such as accuracy and algorithm name.
save_metadata(
    accuracy=accuracy,
    algorithm="Random Forest"
)