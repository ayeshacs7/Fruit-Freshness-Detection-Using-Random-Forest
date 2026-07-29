# 🍎 Fruit Freshness Detection using Machine Learning

A Machine Learning-based Fruit Freshness Detection system that classifies fruit images as **Fresh** or **Rotten** using handcrafted image features and multiple machine learning algorithms. The project compares model performance, automatically selects the best model, and provides predictions through a Streamlit web application.

---

## 📌 Features

- Image preprocessing and handcrafted feature extraction
- Multiple Machine Learning algorithms
  - Random Forest
  - Decision Tree
  - Logistic Regression
- Model comparison
- Automatic best model selection
- Confusion Matrix visualization
- Performance evaluation
- Streamlit web application
- Best model saved for deployment

---

## 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Fruit_Freshness_ML_v2/
│
├── dataset/
├── features/
├── model/
│   └── best_model.pkl
├── screenshots/
├── utils/
├── app.py
├── feature_extraction.py
├── train_model.py
├── requirements.txt
├── comparison_results.csv
├── model_comparison.png
├── random_forest_confusion_matrix.png
├── decision_tree_confusion_matrix.png
├── logistic_regression_confusion_matrix.png
└── README.md
```

---

## 🤖 Machine Learning Algorithms

The following algorithms were trained and evaluated:

- Random Forest
- Decision Tree
- Logistic Regression

---

## 📊 Model Comparison

| Algorithm | Accuracy |
|-----------|----------|
| Random Forest | **87.72%** |
| Decision Tree | 81.58% |
| Logistic Regression | 73.68% |

### 🏆 Best Model

**Random Forest**

Accuracy: **87.72%**

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

## 🖥 Streamlit Application

Run the application using:

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

```bash
python train_model.py
```

---

## 📷 Results

### Model Comparison

![Model Comparison](model_comparison.png)

### Random Forest Confusion Matrix

![Random Forest Confusion Matrix](random_forest_confusion_matrix.png)

### Decision Tree Confusion Matrix

![Decision Tree Confusion Matrix](decision_tree_confusion_matrix.png)

### Logistic Regression Confusion Matrix

![Logistic Regression Confusion Matrix](logistic_regression_confusion_matrix.png)