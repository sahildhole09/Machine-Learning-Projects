#---------------------------------------------------------------
# Importing Required Librares
#---------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#---------------------------------------------------------------
#  Function Name : LoadData
#  Description :   Load the data from CSV
#  Input :         Name of csv file
#  Output :        Data frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def LoadData(Border):

    print(Border)
    print("Step 1 : Loading the dataset")
    print(Border)

    df = pd.read_csv("breast_cancer.csv")

    print("Shape of Dataset : ",df.shape)

    print("\nFirst few records : ")
    print(df.head())

    print("\nDataset loaded successfully")
    print(Border)

    return df

#---------------------------------------------------------------
#  Function Name : AnalyzeData
#  Description :   It performs data analysis
#  Input :         Data Frame
#  Output :        Updated Data Frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def AnalyzeData(Border,df):

    print(Border)
    print("Step 2 : Data Analysis (EDA)")
    print(Border)

    print("Shape of dataset : ",df.shape)

    print("\nColumn names : ",list(df.columns))

    print("\nMissing values per column : ")
    print(df.isnull().sum())

    print("\nClass distribution (species count)")
    print(df["target"].value_counts())

    print("\nStatistical report of dataset : ")
    print(df.describe())

    print(Border)

    return df

#---------------------------------------------------------------
#  Function Name : SeperateData
#  Description :   It performs Seperation of Independent and Dependent Variables
#  Input :         Data Frame
#  Output :        X for Independent Variable and Y for Dependent Variable
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def SeperateData(Border,df):

    print(Border)
    print("Step 3 : Decide Independent & Dependent Variables")
    print(Border)

    X = df.drop("target",axis=1)
    Y = df["target"]

    print("X Shape : ",X.shape)
    print("Y Shape : ",Y.shape)

    print(Border)

    return X,Y

#---------------------------------------------------------------
#  Function Name : SplitData
#  Description :   It performs Splitting activity
#  Input :         Data Frame
#  Output :        4 subsets for training and testing
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def SplitData(Border,X,Y):

    print(Border)
    print("Step 4 : Split the dataset for training and testing")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
    )

    print("X_train : ",X_train.shape)
    print("X_test : ",X_test.shape)

    print("Y_train : ",Y_train.shape)
    print("Y_test : ",Y_test.shape)

    print("\nDataset splitting activity done")

    print(Border)

    return X_train,X_test,Y_train,Y_test

#---------------------------------------------------------------
#  Function Name : ScaleData
#  Description :   It performs Scaling of Independent Variables
#  Input :         Data Frame
#  Output :        4 subsets for training and testing
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def ScaleData(Border,X_train,X_test):

    print(Border)
    print("Step 4 : Scaling the Independent Variables")
    print(Border)

    scalar = StandardScaler()

    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)

    print(Border)

    return scalar,X_train,X_test

#---------------------------------------------------------------
#  Function Name : main
#  Description :   Entry point function
#  Input :         None
#  Output :        None
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def main():
    Border = "-"*70

    df = LoadData(Border) 

    df = AnalyzeData(Border,df)

    X,Y = SeperateData(Border,df)

    X_train,X_test,Y_train,Y_test = SplitData(Border,X,Y)

    scalar,X_train,X_test = ScaleData(Border,X_train,X_test)

if __name__ == "__main__":
    main()