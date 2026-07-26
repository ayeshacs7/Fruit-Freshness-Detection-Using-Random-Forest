from feature_extraction import extract_features
from utils.model_utils import load_model

import os


image = r"D:\Fruit_Freshness_ML\dataset\Apple\Fresh\apple_fresh_001.jpg (4).jpg"
print("Image exists:", os.path.isfile(image))
model = load_model()

features = extract_features(image)

print("Feature Length:", len(features))

prediction = model.predict([features])[0]

print("Prediction:", prediction)