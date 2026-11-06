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
#  Function Name : ElbowMethod
#  Description   : It calculates the WCSS
#  Input         : Scaled Feature
#  Output        : Values of WCSS
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def ElbowMethod(Border,X_scaled):
    print(Border)
    print("Elbow Method")
    print(Border)

    WCSS = []

    for k in range(1,11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
            )
        
        model.fit(X_scaled)

        WCSS.append(model.inertia_)

    print("Values of WCSS : ")
    for i in range(len(WCSS)):
        print(f"{i+1} : {WCSS[i]}")

    return WCSS

#------------------------------------------------------------
#  Function Name : DataVisualization
#  Description   : It visualizes the Elbow Method
#  Input         : WCSS
#  Output        : None
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def DataVisualization(Border,WCSS):
    print(Border)
    print("Visualization of Elbow Method")
    print(Border)

    plt.plot(range(1,11),WCSS,marker="o")
    plt.xlabel("Number of Clusters : k")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

#------------------------------------------------------------
#  Function Name : FinalModel
#  Description   : It performs model training & prediction of clusters
#  Input         : Scaled Feature, dataframe
#  Output        : Dataframe consist of predicted Cluster column
#  Author        : Sahil Ashok Dhole
#------------------------------------------------------------

def FinalModel(Border,X_scaled,df):
    print(Border)
    print("Final Model")
    print(Border)

    model = KMeans(
            n_clusters=4,
            random_state=42,
            n_init=10
            )
    
    clusters = model.fit_predict(X_scaled)

    df["Cluster"] = clusters

    print("Dataset with clusters : ")
    print(df.head(100))

    print(Border)

    return df

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

    WCSS = ElbowMethod(Border,X_scaled)

    DataVisualization(Border,WCSS)

    FinalModel(Border,X_scaled,df)

#------------------------------------------------------------
# Application Starter
#------------------------------------------------------------

if __name__ == "__main__":
    main()