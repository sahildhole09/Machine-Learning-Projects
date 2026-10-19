#---------------------------------------------------------------
# Importing Required Librares
#---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

#---------------------------------------------------------------
#  Function Name : LoadData
#  Description :   Load the data from CSV
#  Input :         Name of csv file
#  Output :        Data frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def LoadData(border,DataPath):

    print(border)
    print("Step 1 : Load the dataset from CSV file")
    print(border)

    df = pd.read_csv(DataPath)

    print(border)
    print("Some entries from dataset : ")
    print(df.head())
    print(border)

    return df

#---------------------------------------------------------------
#  Function Name : CleanData
#  Description :   It performs data cleaning
#  Input :         Data Frame
#  Output :        Updated Data Frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def CleanData(border,df):

    print(border)
    print("Step 2 : Clean the dataset")
    print(border)

    df.dropna(inplace= True)

    print("Shape of dataset : ",df.shape)

    print("Total records : ",df.shape[0])
    print("Total columns : ",df.shape[1])

    print(border)

    return df

#---------------------------------------------------------------
#  Function Name : SeperateData
#  Description :   It performs Seperation of Independent and Dependent Variables
#  Input :         Data Frame
#  Output :        X for Independent Variable and Y for Dependent Variable
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def SeperateData(border,df):

    print(border)
    print("Step 3 : Seperate Independent and Dependent Variables")
    print(border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)

    print(border)
    print("Input columns : ",X.columns.tolist())
    print("Output column : Class")
    print(border)

    return X,Y

#---------------------------------------------------------------
#  Function Name : SplitData
#  Description :   It performs Splitting activity
#  Input :         Data Frame
#  Output :        4 subsets for training and testing
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def SplitData(border,X,Y):

    print(border)
    print("Step 4 : Split the dataset for training and testing")
    print(border)

    X_train,X_test,Y_train,Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
        )

    print(border)
    print("Details of training and testing data : ")

    print("Shape of X_train : ",X_train.shape)
    print("Shape of X_test : ",X_test.shape)

    print("Shape of Y_train : ",Y_train.shape)
    print("Shape of Y_test : ",Y_test.shape)

    print(border)

    return X_train,X_test,Y_train,Y_test

#---------------------------------------------------------------
#  Function Name : ScaleData
#  Description :   It performs Features Scaling
#  Input :         Independent Variables of Training and Testing 
#  Output :        Scalar Object, Scaled Independent Variables of Training and Testing 
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def ScaleData(border,X_train,X_test):

    print(border)
    print("Step 5 : Feature Scaling")
    print(border)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feature Scaling Done")

    print(border)

    return scalar,X_train_scaled,X_test_scaled

#---------------------------------------------------------------
#  Function Name : main
#  Description :   Entry point function
#  Input :         None
#  Output :        None
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def main():
    border = "-"*70

    DataPath = "WinePredictor.csv" 

    df = LoadData(border,DataPath)

    df = CleanData(border,df)

    X,Y = SeperateData(border,df)

    X_train,X_test,Y_train,Y_test = SplitData(border,X,Y)

    scalar,X_train_scaled,X_test_scaled = ScaleData(border,X_train,X_test)

#---------------------------------------------------------------
# Starter
#---------------------------------------------------------------

if __name__ == "__main__":
    main()