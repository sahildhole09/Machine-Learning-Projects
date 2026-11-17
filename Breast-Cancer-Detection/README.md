# Breast Cancer Detection System

A machine learning project that classifies breast-cancer dataset records into **Malignant** or **Benign** classes. It includes multiple classification algorithms and a Streamlit web interface for predictions, dataset exploration, and model-performance comparison.

## 1. Project Overview

The project uses numerical measurements from the breast cancer dataset to train machine learning classification models. The Streamlit application allows users to select a model, enter feature values, and view the predicted class.

The supplied dataset contains **569 records**, **30 input features**, and one target column (`target`).

- `target = 0` — Malignant
- `target = 1` — Benign

## 2. Features

- Interactive Streamlit dashboard.
- Prediction using a selected machine learning model.
- Comparison of model accuracy.
- Confusion matrix and classification report.
- Dataset preview and missing-value overview.
- Target-class distribution chart.
- Prediction probabilities when supported by the selected model.

## 3. Machine Learning Models

The project contains implementations of:

| Model | Technique |
|---|---|
| Decision Tree | Tree-based classification |
| K-Nearest Neighbors (KNN) | Neighbor-based classification |
| Logistic Regression | Linear classification |
| Bagging | Ensemble learning with a Decision Tree base estimator |
| AdaBoost | Boosting ensemble |
| Hard Voting | Combines Logistic Regression, Decision Tree, and KNN predictions |
| Soft Voting | Combines class probabilities from Logistic Regression, Decision Tree, and KNN |
| Random Forest | Tree-based ensemble |

**Note:** The Streamlit application currently loads and compares seven models: Decision Tree, KNN, Logistic Regression, Bagging, AdaBoost, Hard Voting, and Soft Voting. The Random Forest script is included as a separate model file but is not imported into the current Streamlit interface.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## 5. Project Files

| File | Description |
|---|---|
| `breast_cancer.csv` | Dataset used for model training and prediction |
| `BreastCancerDetectionDT.py` | Decision Tree implementation |
| `BreastCancerDetectionKNN.py` | KNN implementation |
| `BreastCancerDetectionLR.py` | Logistic Regression implementation |
| `BreastCancerBagging.py` | Bagging implementation |
| `BreastCancerBoosting.py` | AdaBoost implementation |
| `BreastCancerHardVoting.py` | Hard Voting implementation |
| `BreastCancerSoftVoting.py` | Soft Voting implementation |
| `BreastCancerRandomForest.py` | Random Forest implementation |
| `BreastCancerDetectionStreamlit.py` | Main Streamlit application |
| `requirements.txt` | Python package dependencies |

## 6. Installation

1. Install Python.
2. Download or clone this project.
3. Open a terminal in the project folder.
4. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

5. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 7. How to Run

Keep the CSV file and Python files in the same project folder. Then run:

```bash
streamlit run BreastCancerDetectionStreamlit.py
```

The application will open in your browser. Use the sidebar to navigate between **Dashboard**, **Prediction**, **Model Performance**, and **Dataset**.

## 8. How the Project Works

1. Load the breast cancer dataset.
2. Examine the dataset and target distribution.
3. Separate the input features (`X`) and target (`Y`).
4. Split the data into training and testing sets.
5. Scale the input features using `StandardScaler`.
6. Create and train the selected machine learning models.
7. Evaluate predictions using accuracy, confusion matrix, and classification report.
8. Use the Streamlit interface to make predictions and explore results.

## 9. Important Note

This project is intended for **learning and educational purposes only**. Its predictions are not a medical diagnosis and must not be used to make healthcare decisions. Always consult a qualified healthcare professional for medical advice.

## 10. Author

**Sahil Ashok Dhole**
