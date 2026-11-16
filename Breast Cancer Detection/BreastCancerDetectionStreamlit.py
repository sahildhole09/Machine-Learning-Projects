import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import BreastCancerDetectionDT as dt
import BreastCancerDetectionKNN as knn
import BreastCancerDetectionLR as lr
import BreastCancerBagging as bag
import BreastCancerBoosting as boost
import BreastCancerHardVoting as hard_vote
import BreastCancerSoftVoting as soft_vote

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

st.title("🩺 Breast Cancer Detection System")
st.caption("Machine learning based classification using your existing model functions.")

@st.cache_data
def load_dataset():
    return pd.read_csv("breast_cancer.csv")

@st.cache_resource
def train_all_models():
    df = load_dataset()
    border = "-" * 50
    X, Y = dt.SeperateData(border, df)

    model_modules = {
        "Decision Tree": dt,
        "K-Nearest Neighbors": knn,
        "Logistic Regression": lr,
        "Bagging": bag,
        "AdaBoost": boost,
        "Hard Voting": hard_vote,
        "Soft Voting": soft_vote,
    }

    trained = {}
    for name, module in model_modules.items():
        X_train, X_test, y_train, y_test = module.SplitData(border, X, Y)
        scaler, X_train_scaled, X_test_scaled = module.ScaleData(
            border, X_train, X_test
        )

        if name == "Bagging":
            base_model = module.BaseModelCreation(border)
            model = module.BaggingModelCreation(border, base_model)
        elif name in ["Hard Voting", "Soft Voting"]:
            model_log, model_det, model_knn = module.IndividualModelsCreation(border)
            model = module.VotingModelCreation(
                border, model_log, model_det, model_knn
            )
        else:
            model = module.ModelCreation(border)

        model = module.ModelTraining(border, model, X_train_scaled, y_train)
        y_pred = module.ModelTesting(border, model, X_test_scaled, y_test)
        accuracy, cm, report = module.ModelEvaluating(border, y_test, y_pred)

        trained[name] = {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "report": report,
            "features": list(X.columns),
        }

    return trained

df = load_dataset()

with st.spinner("Preparing your existing ML models..."):
    trained_models = train_all_models()

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction", "Model Performance", "Dataset"]
)
