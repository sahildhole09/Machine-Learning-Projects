#---------------------------------------------------------------
# Importing Required Librares
#---------------------------------------------------------------

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

    DataPath = "customer_churn.csv"

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
    print(df["Churn"].value_counts())

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

    feature_cols = [
        "Age",
        "Tenure",
        "MonthlyCharges",
        "SupportCalls"
        ]
    
    X = df[feature_cols]

    Y = df["Churn"]

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

    model = LogisticRegression(max_iter=1000)

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

    accuracy = accuracy_score(Y_test,Y_pred)
    print("Accuracy of model is : ",accuracy*100,"%")
    print(Border)
  
    print("Confusion Matrix")
    cm = confusion_matrix(Y_test,Y_pred)
    print(cm)
    print(Border)

    print("Classification Report")
    report = classification_report(Y_test,Y_pred)
    print(report)
    print(Border)

    return accuracy,cm,report
#---------------------------------------------------------------
#  Function Name : NewRecord
#  Description :   It predicts the output of new customer 
#  Input :         model
#  Output :        prediction
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def NewRecord(Border,model):

    print(Border)
    print("Prediction of New Customer")
    print(Border)

    age = int(input("Enter your age : "))
    tenure = int(input("\nEnter your tenure : "))
    charges = int(input("\nEnter your charges : "))
    calls = int(input("\nEnter your calls : "))

    new_customer = pd.DataFrame({
    "Age" : [age],
    "Tenure" : [tenure],
    "MonthlyCharges" : [charges],
    "SupportCalls" : [calls]
    })

    prediction = model.predict(new_customer)

    if prediction[0] == 1:
        print("\nPrediction : Customer is likely to churn")
    else:
        print("\nPrediction : Customer is unlikely to churn")

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

    accuracy,cm,report = EvaluateModel(Border,Y_test,Y_pred)

    NewRecord(Border,model)

#---------------------------------------------------------------
# Starter
#---------------------------------------------------------------

if __name__ == "__main__":
    main()
