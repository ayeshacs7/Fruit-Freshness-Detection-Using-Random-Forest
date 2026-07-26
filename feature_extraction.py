# ============================================================
# Fruit Freshness Detection using Machine Learning
# Author: Ayesha Shafique
# Description:
# This module extracts image features that are later used
# to train the machine learning model for classifying
# fruits as Fresh or Rotten.
# ============================================================
# Import required libraries
import cv2
import numpy as np
# ------------------------------------------------------------
# Function: extract_features()
# Purpose:
# Reads an image and extracts important visual features
# for machine learning classification.
# ------------------------------------------------------------
def extract_features(image_path):
    # Reading image from disk
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    # RGB Mean
    rgb_mean = np.mean(image, axis=(0, 1))

    # HSV Mean
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mean = np.mean(hsv, axis=(0, 1))

    # Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness & Contrast
    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Edge Density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Color Histogram
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [8, 8],
        [0, 180, 0, 256]
    )
# Normalize the histogram so that feature values remain
# consistent even if image brightness or size varies.
    hist = cv2.normalize(hist, hist).flatten()
# Combine all extracted features into a single feature vector.
# This feature vector will be used by the Random Forest model
# during both training and prediction.
    features = np.hstack([
        rgb_mean,
        hsv_mean,
        brightness,
        contrast,
        edge_density,
        hist
    ])
# Return the complete feature vector to the machine
# learning pipeline.
    return features