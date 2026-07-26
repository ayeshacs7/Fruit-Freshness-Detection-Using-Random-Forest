import os

from feature_extraction import extract_features
from utils.model_utils import load_model


CLASS_NAMES = {
    0: "Fresh Apple",
    1: "Rotten Apple",
    2: "Fresh Banana",
    3: "Rotten Banana",
    4: "Fresh Strawberry",
    5: "Rotten Strawberry"
}


model = load_model()

print("=" * 50)
print("Fruit Freshness Prediction")
print("=" * 50)

image_path = input("\nEnter image path: ").strip().strip('"')

if not os.path.isfile(image_path):
    print(f"\nError: File not found:\n{image_path}")
    exit()

if not os.path.exists(image_path):
    print("\nError: Image not found.")
    exit()

features = extract_features(image_path)

prediction = model.predict([features])[0]

print("\nPrediction:")
print(CLASS_NAMES[prediction])