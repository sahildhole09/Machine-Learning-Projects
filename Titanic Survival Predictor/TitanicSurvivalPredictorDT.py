#---------------------------------------------------------------
# Importing Required Libraries
#---------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

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
#  Date :          16/08/2026
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
#  Date :          16/08/2026
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
#  Function Name : main
#  Description :   Entry point function
#  Input :         None
#  Output :        None
#  Author :        Sahil Ashok Dhole
#  Date :          16/08/2026
#---------------------------------------------------------------

def main():
    df = LoadData("MarvellousTitanicDataset.csv")

    df = PreprocessData(df)

    X_train,X_test,Y_train,Y_test = SplitData(df)

#---------------------------------------------------------------
# Starter
#---------------------------------------------------------------
    
if __name__ == "__main__":
    main()