# ==========================================================
# Project : Fruit Freshness Detection using Machine Learning
# Author  : Ayesha Shafiq
# File    : train_model.py
# Purpose :
# This module loads the dataset, trains multiple
# machine learning models, compares their
# performance, selects the best-performing model,
# and saves it for deployment.
# ==========================================================
# Import required libraries.
import os
import joblib
# Import plotting and data analysis libraries.
import matplotlib.pyplot as plt
import pandas as pd
# Import function to split the dataset.
from sklearn.model_selection import train_test_split
# Import custom dataset loader and model utilities.
from utils.dataset_loader import load_dataset
from utils.model_utils import save_model, save_metadata
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
# ==========================================================
# Save Confusion Matrix
# ==========================================================
def save_confusion_matrix(y_true, y_pred, algorithm):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    ax.set_title(f"{algorithm} Confusion Matrix")

    filename = f"{algorithm.lower().replace(' ', '_')}_confusion_matrix.png"

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {filename}")
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
# Store the performance of all machine learning models.
results = []
# Display a message indicating the start of model training.
# ==========================================================
# Train Random Forest Model
# ==========================================================
def train_random_forest(X_train, X_test, y_train, y_test):

    print("\nTraining Random Forest Model...")

    # Create the classifier.
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Train the model.
    model.fit(X_train, y_train)

    # Predict testing data.
    predictions = model.predict(X_test)
    save_confusion_matrix(
    y_test,
    predictions,
    "Random Forest"
)
    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

    print(classification_report(y_test, predictions))

    return (
        model,
        accuracy,
        precision,
        recall,
        f1,
        predictions
    )
rf_model, rf_accuracy, rf_precision, rf_recall, rf_f1, rf_predictions = train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
)

results.append({
    "Algorithm": "Random Forest",
    "Accuracy": rf_accuracy,
    "Precision": rf_precision,
    "Recall": rf_recall,
    "F1-Score": rf_f1,
    "Model": rf_model
})
# ==========================================================
# Train Decision Tree Model
# ==========================================================
def train_decision_tree(X_train, X_test, y_train, y_test):

    print("\nTraining Decision Tree Model...")

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    save_confusion_matrix(
    y_test,
    predictions,
    "Decision Tree"
)
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\nDecision Tree Accuracy: {accuracy * 100:.2f}%\n")

    print(classification_report(y_test, predictions))

    return (
        model,
        accuracy,
        precision,
        recall,
        f1,
        predictions
    )
dt_model, dt_accuracy, dt_precision, dt_recall, dt_f1, dt_predictions = train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test
)

results.append({
    "Algorithm": "Decision Tree",
    "Accuracy": dt_accuracy,
    "Precision": dt_precision,
    "Recall": dt_recall,
    "F1-Score": dt_f1,
    "Model": dt_model
})
# ==========================================================
# Train Logistic Regression Model
# ==========================================================
def train_logistic_regression(X_train, X_test, y_train, y_test):

    print("\nTraining Logistic Regression Model...")

    # Create the classifier.
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    # Train the model.
    model.fit(X_train, y_train)

    # Predict testing data.
    predictions = model.predict(X_test)
    save_confusion_matrix(
    y_test,
    predictions,
    "Logistic Regression"
)
    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\nLogistic Regression Accuracy: {accuracy * 100:.2f}%\n")

    print(classification_report(y_test, predictions))

    return (
        model,
        accuracy,
        precision,
        recall,
        f1,
        predictions
    )
# Train Logistic Regression model.
lr_model, lr_accuracy, lr_precision, lr_recall, lr_f1, lr_predictions = train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test
)

# Store Logistic Regression results.
results.append({
    "Algorithm": "Logistic Regression",
    "Accuracy": lr_accuracy,
    "Precision": lr_precision,
    "Recall": lr_recall,
    "F1-Score": lr_f1,
    "Model": lr_model
})

print(results)
print("\n==============================")
print(" Model Comparison")
print("==============================")

for result in results:
    print(f"{result['Algorithm']}: {result['Accuracy']:.4f}")

    # Create a DataFrame containing model comparison results.
comparison_df = pd.DataFrame(results)

# Remove the model object before saving the CSV file.
comparison_df = comparison_df.drop(columns=["Model"])

# Save the comparison results.
comparison_df.to_csv(
    "comparison_results.csv",
    index=False
)

print("\nComparison results saved as comparison_results.csv")

# Create a bar chart comparing model accuracies.
plt.figure(figsize=(8, 5))

plt.bar(
    comparison_df["Algorithm"],
    comparison_df["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Model comparison graph saved successfully!")

# ==========================================================
# Select Best Model
# ==========================================================
best_model = max(
    results,
    key=lambda x: x["Accuracy"]
)

print("\n==============================")
print(" Best Model")
print("==============================")

print(f"Accuracy  : {best_model['Accuracy']:.4f}")

# Save the best-performing model.
joblib.dump(
    best_model["Model"],
    "model/best_model.pkl"
)
print(f"Algorithm : {best_model['Algorithm']}")
print("\nBest model saved successfully!")
