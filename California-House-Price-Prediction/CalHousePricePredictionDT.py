#---------------------------------------------------------------
# Importing Required Librares
#---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_squared_error,r2_score)

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

    model = DecisionTreeRegressor(random_state=42)

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
  
    R2_SCORE = r2_score(Y_test,Y_pred)
    print("R2 Score : ",R2_SCORE)
    print(Border)

    return MSE, R2_SCORE

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

    MSE, R2_SCORE = EvaluateModel(Border,Y_test,Y_pred)

#---------------------------------------------------------------
# Application Starter
#---------------------------------------------------------------

if __name__ == "__main__":
    main()