#---------------------------------------------------------------
# Importing Required Librares
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

def LoadData(Border):

    print(Border)
    print("Step 1 : Loading the Dataset ->")
    print(Border)

    Dataset = "WinePredictor.csv"

    df = pd.read_csv(Dataset)

    print("Initial 5 entries from dataset are : \n")
    print(df.head())

    print(Border)
    print("Dataset gets loaded successfully")
    print(Border)

    return df 

#---------------------------------------------------------------
#  Function Name : AnalyseData
#  Description :   It performs data analysis
#  Input :         Data Frame
#  Output :        Updated Data Frame
#  Author :        Sahil Ashok Dhole
#---------------------------------------------------------------

def AnalyseData(Border,df):
    
    print(Border)
    print("Step 2 : Data Analysis (EDA) ->")
    print(Border)

    print("Shape of Dataset : \n",df.shape)

    print("\nList of Column Names in Dataset : \n",df.columns)

    print("\nMissing values per Column : \n",df.isnull().sum())

    print("\nClass distribution (Class Count) : \n",df["Class"].value_counts())

    print("\nStatistical Report of Dataset : \n\n",df.describe)

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
    print("Step 3 : Seperating the Dataset in X & Y -> ")
    print(Border)

    print("Decide Independent & Dependent Variables : \n")

    features = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
    ]

    X = df[features]
    Y = df["Class"]

    print("Shape of Independent Variables (X) : ",X.shape)
    print("Shape of Dependent Variables (Y) : ",Y.shape)

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
    print("Step 4 : Splitting the Dataset -> ")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=42)

    print("Dataset splitting activity done :\n")    

    print("X_train : ",X_train.shape)  
    print("X_test : ",X_test.shape)     

    print("Y_train : ",Y_train.shape)  
    print("Y_test : ",Y_test.shape) 

    print(Border)

    return X_train,X_test,Y_train,Y_test

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

    df = AnalyseData(Border,df)

    X,Y = SeperateData(Border,df)

    SplitData(Border,X,Y)

#---------------------------------------------------------------
# Starter
#---------------------------------------------------------------

if __name__ == "__main__":
    main()