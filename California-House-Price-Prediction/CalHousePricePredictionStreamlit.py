#---------------------------------------------------------------
# Importing Required Librares
#---------------------------------------------------------------

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#---------------------------------------------------------------

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 California House Price Prediction")
st.write("Predict house prices using Machine Learning.")

@st.cache_data
def DataLoading():

    df = pd.read_csv("california_housing.csv")
    return df

df = DataLoading()

st.sidebar.title("Project Navigation")

choice = st.sidebar.selectbox(
    "Select Operation",
    [
        "Home",
        "Data Analysis",
        "Data Visualization",
        "Model Training",
        "New Prediction"
    ]
)

X = df.drop("target", axis=1)
Y = df["target"]

model_name = st.sidebar.selectbox(
    "Select ML Algorithm",
    [
        "Decision Tree",
        "Linear Regression"
    ]
)

@st.cache_resource
def TrainModel(model_name):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.5,
        random_state=42
    )

    if model_name == "Decision Tree":
        model = DecisionTreeRegressor(
            random_state=42
        )

    else:
        model = LinearRegression()

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    return model, X_test, Y_test, Y_pred

model, X_test, Y_test, Y_pred = TrainModel(model_name)

if choice == "Home":

    st.header("Welcome to California House Price Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))

    col2.metric("Total Features", len(X.columns))

    col3.metric("Selected Model", model_name)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

elif choice == "Data Analysis":

    st.header("Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Column Names")
        st.write(list(df.columns))

    with col2:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

    st.subheader("Statistical Report")

    st.dataframe(df.describe(), use_container_width=True)

elif choice == "Data Visualization":

    st.header("Data Visualization")

    st.subheader("1. House Price Distribution")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        df["target"],
        bins=40,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("House Price (units of $100,000)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("2. Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. Median Income vs House Price")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=df.sample(min(3000, len(df)), random_state=42),
        x="MedInc",
        y="target",
        alpha=0.5,
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. Actual vs Predicted House Prices")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        x=Y_test,
        y=Y_pred,
        alpha=0.5,
        ax=ax
    )

    ax.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        color="red",
        linestyle="--"
    )

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("5. Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.barplot(
            data=importance,
            x="Importance",
            y="Feature",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.info(
            "Feature importance visualization is available "
            "for Decision Tree only."
        )

        