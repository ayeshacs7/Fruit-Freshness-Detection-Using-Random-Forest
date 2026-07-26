# Import required libraries.
import os
import numpy as np
# Import the feature extraction function.
from feature_extraction import extract_features
# Root directory containing all fruit image folders.
DATASET_PATH = "dataset"
# Map each fruit category to a unique numeric label.
# These labels are used by the machine learning model.
CLASSES = {
    "Apple/Fresh": 0,"Apple/Rotten": 1,"Banana/Fresh": 2,"Banana/Rotten": 3,
    "Strawberry/Fresh": 4,"Strawberry/Rotten": 5
}
  # Load the dataset, extract image features,
    # and prepare feature vectors with their labels.
def load_dataset():
# Store extracted feature vectors
    X = []
# Store corresponding class labels
    y = []
# Display a message indicating that
# the dataset loading process has started.
    print("Loading Dataset...\n")
# Iterate through every fruit category
# defined in the class dictionary.
    for class_name, label in CLASSES.items():
# Separate the fruit type and freshness status.
        fruit, condition = class_name.split("/")
# Construct the complete folder path
# containing images of the current class.
        folder = os.path.join(DATASET_PATH, fruit, condition)
# Count successfully processed images
# for the current category.
        image_count = 0
# Process every image available
# inside the current folder.
        for image in os.listdir(folder):
# Create the complete path
# of the current image.
            image_path = os.path.join(folder, image)
# Extract handcrafted image features.
            features = extract_features(image_path)
# Store valid feature vectors
# along with their class labels.
            if features is not None:
# Display the total number of processed
# images for the current category.
                X.append(features)
                y.append(label)
                image_count += 1
# Convert Python lists into NumPy arrays
# before returning them for model training.
        print(f"{class_name:<20} : {image_count} images")
# Convert Python lists into NumPy arrays
# before returning them for model training.
    return np.array(X), np.array(y)