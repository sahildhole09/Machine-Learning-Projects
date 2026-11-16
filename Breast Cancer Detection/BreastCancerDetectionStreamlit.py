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

if page == "Dashboard":
    st.subheader("Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Input Features", len(df.columns) - 1)
    c3.metric("Models Available", len(trained_models))

    st.subheader("Target Class Distribution")
    counts = df["target"].map({0: "Malignant", 1: "Benign"}).value_counts()
    st.bar_chart(counts)

    st.info("Select Prediction to enter feature values and get a model result.")

elif page == "Prediction":
    st.subheader("Make a Prediction")
    selected_model = st.selectbox("Select Model", list(trained_models.keys()))
    model_info = trained_models[selected_model]

    st.write("Enter the 30 feature values:")
    values = {}
    features = model_info["features"]
    defaults = df[features].median()

    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            values[feature] = st.number_input(
                feature,
                value=float(defaults[feature]),
                format="%.5f",
                key=f"{selected_model}_{feature}"
            )

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([values], columns=features)
        scaled_input = model_info["scaler"].transform(input_df)
        prediction = model_info["model"].predict(scaled_input)[0]

        if prediction == 0:
            st.error("Prediction: Malignant")
        else:
            st.success("Prediction: Benign")

        if hasattr(model_info["model"], "predict_proba"):
            probabilities = model_info["model"].predict_proba(scaled_input)[0]
            probability_df = pd.DataFrame(
                {"Class": ["Malignant", "Benign"], "Probability": probabilities}
            ).set_index("Class")
            st.subheader("Prediction Probability")
            st.bar_chart(probability_df)

        st.warning(
            "This application is for educational purposes only. "
            "It is not a medical diagnosis. Consult a qualified healthcare professional."
        )
