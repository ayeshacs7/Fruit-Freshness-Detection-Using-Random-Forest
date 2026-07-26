import os
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from feature_extraction import extract_features


# -----------------------------
# Configuration
# -----------------------------

DATASET_PATH = "dataset"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# -----------------------------
# Class Labels
# -----------------------------

classes = {
    "Apple/Fresh": 0,
    "Apple/Rotten": 1,
    "Banana/Fresh": 2,
    "Banana/Rotten": 3,
    "Strawberry/Fresh": 4,
    "Strawberry/Rotten": 5
}


X = []
y = []

print("Loading Dataset...\n")

for class_name, label in classes.items():

    fruit, condition = class_name.split("/")

    folder = os.path.join(DATASET_PATH, fruit, condition)

    for image in os.listdir(folder):

        image_path = os.path.join(folder, image)

        features = extract_features(image_path)

        if features is not None:

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total Samples : {len(X)}")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE
    ),

    "SVM": SVC(kernel="rbf")
}

print("\nModel Comparison")
print("-" * 45)

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name:<20} : {accuracy*100:.2f}%")

best_model = max(results, key=results.get)

print("\n" + "=" * 45)
print(f"Best Model : {best_model}")
print(f"Accuracy   : {results[best_model]*100:.2f}%")
print("=" * 45)