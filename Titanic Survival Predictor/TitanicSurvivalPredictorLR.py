#---------------------------------------------------------------
# Importing Required Libraries
#---------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

#---------------------------------------------------------------
#  Function Name : LoadData
#  Description :   Load the data from CSV
#  Input :         Name of csv file
#  Output :        Data frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def LoadData(filename):
    df = pd.read_csv(filename)

    print("Dataset loaded successfully...")
    print(df.head())

    return df

#---------------------------------------------------------------
#  Function Name : PreprocessData
#  Description :   It performs data analysis
#  Input :         Data Frame
#  Output :        Updated Data Frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def PreprocessData(df):
    df = df.drop([
        "Passengerid",
        "zero",
        "name"
    ],
    errors = "ignore"
    )

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    print(df.head())

    print("Data preprocessing completed")

    return df

#---------------------------------------------------------------
#  Function Name : SplitData
#  Description :   It performs Splitting activity
#  Input :         Data Frame
#  Output :        4 subsets for training and testing
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def SplitData(df):
    X = df.drop("Survived",axis = 1)
    Y = df["Survived"]

    X_train,X_test,Y_train,Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Data splitting completed successfully...")
    return X_train,X_test,Y_train,Y_test

#---------------------------------------------------------------
#  Function Name : TrainModel
#  Description :   It performs model training
#  Input :         Training features and labels
#  Output :        Trained model
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def TrainModel(X_train,Y_train):
    model = LogisticRegression(max_iter=1000)

    model = model.fit(X_train,Y_train)

    print("Trained model successfully...")

    return model

#---------------------------------------------------------------
#  Function Name : EvaluateModel
#  Description :   It performs model testing
#  Input :         model,testing data(features,labels)
#  Output :        none
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def EvaluateModel(model,X_test,Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Accuracy : ",accuracy)

    print("Confusion Matrix : ")

    print(confusion_matrix(Y_test,Y_pred))

    print("Classification Report : ")

    print(classification_report(Y_test,Y_pred))

#---------------------------------------------------------------
#  Function Name : PreserveModel
#  Description :   It performs model preservation into .pkl file
#  Input :         model
#  Output :        None
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def PreserveModel(model,filename):
    joblib.dump(model,filename)

    print("Model preserved with name : ",filename)

#---------------------------------------------------------------
#  Function Name : main
#  Description :   Entry point function
#  Input :         None
#  Output :        None
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def main():
    df = LoadData("MarvellousTitanicDataset.csv")

    df = PreprocessData(df)

    X_train,X_test,Y_train,Y_test = SplitData(df)

    model = TrainModel(X_train,Y_train)

    EvaluateModel(model,X_test,Y_test)

    PreserveModel(model,"MarvellousTitanic.pkl")

#---------------------------------------------------------------
# Starter
#---------------------------------------------------------------
    
if __name__ == "__main__":
    main()