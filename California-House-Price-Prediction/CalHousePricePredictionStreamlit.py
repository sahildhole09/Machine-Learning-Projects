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

