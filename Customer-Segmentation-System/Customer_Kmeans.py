#------------------------------------------------------------
# Importing required libraries
#------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#------------------------------------------------------------
#  Function Name : LoadData
#  Description   : Load the data from CSV
#  Input         : Name of csv file
#  Output        : Data frame
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def LoadData(Border):
    print(Border)
    print("Loading the dataset of customer details of mall")
    print(Border)

    Dataset = "Mall_Customers.csv"

    df = pd.read_csv(Dataset)

    print("Initial 5 entries from dataset are : \n")
    print(df.head())

    print("\nDataset gets loaded successfully")

    return df

#------------------------------------------------------------
#  Function Name : AnalyseData
#  Description   : Load the data from CSV
#  Input         : Name of csv file
#  Output        : Data frame
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def AnalyseData(Border,df):
    print(Border)
    print("Data Analysis (EDA)")
    print(Border)

    print("Shape of Dataset : \n",df.shape)

    print("\nList of Column Names in Dataset : \n",df.columns)

    print("\nMissing values per Column : \n",df.isnull().sum())

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["AnnualIncome"] = df["AnnualIncome"].fillna(df["AnnualIncome"].median())

    df["SpendingScore"] = df["SpendingScore"].fillna(df["SpendingScore"].median())

    print("\nStatistical Report of Dataset : \n\n",df.describe)

    print("\nData analyzed successfully")

    return df

#------------------------------------------------------------
#  Function Name : FeatureSelection
#  Description   : Selecting the features for model training & testing
#  Input         : Dataset
#  Output        : Selected Features
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def FeatureSelection(Border,df):
    print(Border)
    print("Feature Selection")
    print(Border)

    X = df[["AnnualIncome","SpendingScore"]]

    print("Selected features : ")
    print(X.head())

    print("\nFeature selection successfully done")

    return X

#------------------------------------------------------------
#  Function Name : FeatureScaling
#  Description   : It scale the features
#  Input         : Feature
#  Output        : Scaled Feature
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def FeatureScaling(Border,X):
    print(Border)
    print("Feature Scaling")
    print(Border)

    scalar = StandardScaler()

    X_scaled = scalar.fit_transform(X)

    print("Scaled Data : ")
    print(X_scaled[:5])

    print("\nFeature scaling successfully done")

    return X_scaled

#------------------------------------------------------------
#  Function Name : main
#  Description   : Entry point function
#  Input         : None
#  Output        : None
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def main():
    Border = "-"*60

    df = LoadData(Border)

    df = AnalyseData(Border,df)

    X = FeatureSelection(Border,df)

    X_scaled = FeatureScaling(Border,X)

#------------------------------------------------------------
# Application Starter
#------------------------------------------------------------

if __name__ == "__main__":
    main()