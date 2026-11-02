#---------------------------------------------------------------
# Importing Required Librares
#---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,root_mean_squared_error,r2_score)
import joblib

#---------------------------------------------------------------
#  Function Name : DataLoading
#  Description :   Load the data from CSV
#  Input :         Name of csv file
#  Output :        Data frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def DataLoading(Border):
    print(Border)
    print("Loading the dataset")
    print(Border)

    DataPath = "california_housing.csv"

    df = pd.read_csv(DataPath)

    print("Dataset loaded successfully")
    print("Initial entries from dataset are : ")
    print(df.head())

    return df

#---------------------------------------------------------------
#  Function Name : DataAnalysis
#  Description :   It performs data analysis
#  Input :         Data Frame
#  Output :        Updated Data Frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def DataAnalysis(Border,df):
    print(Border)
    print("Data Analysis (EDA)")
    print(Border)

    print("Shape of dataset : ",df.shape)

    print("Column names : ",list(df.columns))

    print("Missing values per column : ")
    print(df.isnull().sum())

    print("Class distribution (species count)")
    print(df["target"].value_counts())

    print("Correlation : ")
    print(df.corr())

    print("Statistical report of dataset : ")
    print(df.describe())

    return df

#---------------------------------------------------------------
#  Function Name : DataSeperate
#  Description :   It performs Seperation of Independent and Dependent Variables
#  Input :         Data Frame
#  Output :        X for Independent Variable and Y for Dependent Variable
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def DataSeperate(Border,df):
    print(Border)
    print("Decide Independent & Dependent Variables")
    print(Border)

    X = df.drop("target",axis=1)
    Y = df["target"]

    print("X Shape : ",X.shape)
    print("Y Shape : ",Y.shape)

    return X,Y

#---------------------------------------------------------------
#  Function Name : DataSplitting
#  Description :   It performs Splitting activity
#  Input :         Data Frame
#  Output :        4 subsets for training and testing
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def DataSplitting(Border,X,Y):
    print(Border)
    print("Split the dataset for training and testing")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=42)

    print("Dataset splitting activity done")

    print("X : ",X.shape)
    print("Y : ",Y.shape)

    print("X_train : ",X_train.shape)   
    print("X_test : ",X_test.shape)  

    print("Y_train : ",Y_train.shape)  
    print("Y_test : ",Y_test.shape)  

    return X_train,X_test,Y_train,Y_test

#---------------------------------------------------------------
#  Function Name : CreateModel
#  Description :   It performs model creation
#  Input :         None
#  Output :        Created model
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def CreateModel(Border):
    print(Border)
    print("Built the model")
    print(Border)

    model = LinearRegression()

    print("Model gets created successfully")

    return model

#---------------------------------------------------------------
#  Function Name : TrainModel
#  Description :   It performs model training
#  Input :         Model, Training features and labels
#  Output :        Trained model
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def TrainModel(Border,model,X_train,Y_train):
    print(Border)
    print("Train the model")
    print(Border)

    model = model.fit(X_train,Y_train)

    print("Model trained successfully")

    return model

#---------------------------------------------------------------
#  Function Name : TestModel
#  Description :   It performs model testing
#  Input :         Model, Testing features and labels
#  Output :        Predicted Output
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def TestModel(Border,model,X_test,Y_test):
    print(Border)
    print("Test the model")
    print(Border)

    Y_pred = model.predict(X_test)

    print("Model testing done")

    print("Expected answers : ")
    print(Y_test)

    print("Predicted answers : ")
    print(Y_pred)

    return Y_pred

#---------------------------------------------------------------
#  Function Name : EvaluateModel
#  Description :   It calculates accuracy, confusion matrix, classification report
#  Input :         Expected and Actual Output
#  Output :        Accuracy, Confusion Matrix, Classification Report
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def EvaluateModel(Border,Y_test,Y_pred):
    print(Border)
    print("Evaluate the model performance")
    print(Border)

    MSE = mean_squared_error(Y_test,Y_pred)
    print("Mean Squared Error : ",MSE)
    print(Border)

    RMSE = root_mean_squared_error(Y_test,Y_pred)
    print("Root Mean Squared Error : ",RMSE)
    print(Border)
  
    R2_SCORE = r2_score(Y_test,Y_pred)
    print("R2 Score : ",R2_SCORE)
    print(Border)

    return MSE, RMSE, R2_SCORE

#---------------------------------------------------------------
#  Function Name : NewPrediction
#  Description :   It predicts the new house price
#  Input :         Trained Model
#  Output :        New House Price Predicted Result 
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def NewPrediction(Border,model):
    print(Border)
    print("Prediction Price of New House")
    print(Border)

    new_prediction = pd.DataFrame({
        "MedInc" : [3.5000],
        "HouseAge" : [35.0],
        "AveRooms" : [5.2000],
        "AveBedrms" : [1.0500],
        "Population" : [1200.0],
        "AveOccup" : [2.8000],
        "Latitude" : [37.80],
        "Longitude" : [-122.24]
        })
    
    new_pred = model.predict(new_prediction)

    print("Predicted House Price : ",new_pred)

    print(Border)

#---------------------------------------------------------------
#  Function Name : PreserveModel
#  Description :   It performs model preservation into .pkl file 
#  Input :         Trained Model
#  Output :        None
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def PreserveModel(Border,model,filename):
    print(Border)
    print("Preserve the Model")
    print(Border)

    joblib.dump(model,filename)

    print("Model preserved with name : ",filename)
    print(Border)

#---------------------------------------------------------------
#  Function Name : main
#  Description :   Entry point function
#  Input :         None
#  Output :        None
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def main():
    Border = "-"*50

    df = DataLoading(Border)

    df = DataAnalysis(Border,df)

    X,Y = DataSeperate(Border,df)

    X_train,X_test,Y_train,Y_test = DataSplitting(Border,X,Y)

    model = CreateModel(Border)

    model = TrainModel(Border,model,X_train,Y_train)

    Y_pred = TestModel(Border,model,X_test,Y_test)

    MSE, RMSE, R2_SCORE = EvaluateModel(Border,Y_test,Y_pred)

    NewPrediction(Border,model)

    PreserveModel(Border,model,"CalHousePricePredLRModel.pkl")

#---------------------------------------------------------------
# Application Starter
#---------------------------------------------------------------

if __name__ == "__main__":
    main()