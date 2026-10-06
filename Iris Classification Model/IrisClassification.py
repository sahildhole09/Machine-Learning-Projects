#---------------------------------------------------------------
# Importing Required Librares
#---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1 : Load Data

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

    DataPath = "iris.csv"

    df = pd.read_csv(DataPath)        # df -> DataFrame

    print("Dataset loaded successfully")
    print("Initial entries from dataset are : ")
    print(df.head())

    return df

# Step 2 : Data Analysis

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
    print(df["species"].value_counts())

    print("Statistical report of dataset : ")
    print(df.describe())

    return df

# Step 3 : Seperate Independent and Dependent Variables

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

    # X : Independent Variable / Features
    # Y : Dependent Variable / Labels

    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
        ]

    X = df[feature_cols]
    Y = df["species"]

    print("X Shape : ",X.shape)
    print("Y Shape : ",Y.shape)

    return X,Y

# Step 4 : Data Visualization

#---------------------------------------------------------------
#  Function Name : DataVisualization
#  Description :   It is used to visualize the dataset
#  Input :         Data Frame
#  Output :        Scatter Plot of Petal Length & Petal Width
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def DataVisualization(Border,df):
    print(Border)
    print("Visualization of Dataset")
    print(Border)

    # Scatter Plot
    plt.figure(figsize=(7,5))

    for sp in df["species"].unique():
        temp = df[df["species"] == sp]
        plt.scatter(temp["petal length (cm)"], temp["petal width (cm)"], label = sp)

    plt.title("Iris Classification Case Study")

    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")

    plt.legend()
    plt.grid()
    plt.show()

# Step 5 : Splitting the data

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

    print("X : ",X.shape)     # (150,4)
    print("Y : ",Y.shape)     # (150,)

    print("X_train : ",X_train.shape)   # (75,4)
    print("X_test : ",X_test.shape)     # (75,4)

    print("Y_train : ",Y_train.shape)   # (75,)
    print("Y_test : ",Y_test.shape)     # (75,)

    return X_train,X_test,Y_train,Y_test

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

    DataVisualization(Border,df)

    X_train,X_test,Y_train,Y_test = DataSplitting(Border,X,Y)

if __name__ == "__main__":
    main()