# 🌸 Iris Classification using Decision Tree

## 📌 Project Overview

This project demonstrates a **Machine Learning classification case study** using the **Iris dataset** and a **Decision Tree Classifier**.

The program follows a complete machine learning workflow:

**Data Loading → Data Analysis → Feature Selection → Data Visualization → Data Splitting → Model Creation → Model Training → Model Testing → Model Evaluation**

The implementation is written in **Python** using the **procedural programming paradigm** and is divided into separate functions for each stage of the machine learning process.

---

## 🎯 Objective

The objective of this case study is to build a machine learning model that can classify Iris flowers into their respective species based on their sepal and petal measurements.

The model predicts one of the following three classes:

- `setosa`
- `versicolor`
- `virginica`

---

## 📊 Dataset

The project uses the **Iris dataset**, which contains measurements of Iris flowers.

### Features

| Feature | Description |
|---|---|
| `sepal length (cm)` | Length of the sepal |
| `sepal width (cm)` | Width of the sepal |
| `petal length (cm)` | Length of the petal |
| `petal width (cm)` | Width of the petal |

### Target

```text
species
```

Possible values:

```text
setosa
versicolor
virginica
```

The dataset contains **150 records and 4 independent features**.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – Data loading and analysis
- **Matplotlib** – Data visualization
- **Seaborn** – Visualization support
- **Scikit-learn** – Machine learning model and evaluation

### Machine Learning Algorithm

**Decision Tree Classifier**

```python
DecisionTreeClassifier(max_depth=5)
```

---

## 📁 Project Structure

```text
Iris-Classification-Decision-Tree/
│
├── iris.csv
├── Iris_Decision_Tree.py
├── README.md
└── requirements.txt
```

---

## 🔄 Machine Learning Workflow

### 1. Data Loading

The dataset is loaded from the CSV file using Pandas.

```python
df = pd.read_csv("iris.csv")
```

The first few records are displayed to verify that the dataset has been loaded successfully.

---

### 2. Data Analysis

Basic Exploratory Data Analysis (EDA) is performed.

The program checks:

- Dataset shape
- Column names
- Missing values
- Class distribution
- Statistical summary

Example:

```python
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df["species"].value_counts())
print(df.describe())
```

---

### 3. Feature and Target Separation

The independent and dependent variables are separated.

**Independent variables (X):**

```text
sepal length
sepal width
petal length
petal width
```

**Dependent variable (Y):**

```text
species
```

---

### 4. Data Visualization

A scatter plot is created using:

- Petal Length
- Petal Width

Different Iris species are displayed separately to visualize their distribution.

This helps understand how the different classes are distributed in the feature space.

---

### 5. Train-Test Split

The dataset is divided into training and testing datasets.

```python
train_test_split(
    X,
    Y,
    test_size=0.5,
    random_state=42
)
```

The current implementation uses:

- **50% Training Data**
- **50% Testing Data**

Therefore:

```text
Training samples = 75
Testing samples  = 75
```

---

### 6. Model Creation

A Decision Tree Classifier is created with a maximum depth of 5.

```python
model = DecisionTreeClassifier(max_depth=5)
```

The `max_depth` parameter limits the depth of the tree and helps control model complexity.

---

### 7. Model Training

The model is trained using the training dataset.

```python
model.fit(X_train, Y_train)
```

The model learns the relationship between the flower measurements and their corresponding species.

---

### 8. Model Testing

The trained model predicts the species of flowers from the test dataset.

```python
Y_pred = model.predict(X_test)
```

The program displays:

- Expected output
- Predicted output

---

### 9. Model Evaluation

The model is evaluated using three important metrics.

#### Accuracy

```python
accuracy_score(Y_test, Y_pred)
```

Accuracy represents the percentage of correctly classified samples.

#### Confusion Matrix

```python
confusion_matrix(Y_test, Y_pred)
```

The confusion matrix shows the number of correct and incorrect predictions for each species.

#### Classification Report

```python
classification_report(Y_test, Y_pred)
```

The classification report provides:

- Precision
- Recall
- F1-score
- Support

---

## 📈 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct positive predictions |
| Recall | Ability to identify actual classes |
| F1-Score | Balance between precision and recall |
| Confusion Matrix | Detailed prediction results |

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Open the Project

```bash
cd Iris-Classification-Decision-Tree
```

### Step 3: Install Required Libraries

```bash
pip install pandas matplotlib seaborn scikit-learn
```

Or:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Program

```bash
python Iris_Decision_Tree.py
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
matplotlib
seaborn
scikit-learn
```

---

## 🖥️ Expected Output

The program displays:

```text
--------------------------------------------------
Loading the dataset
--------------------------------------------------
Dataset loaded successfully

Initial entries from dataset are :
...

--------------------------------------------------
Data Analysis (EDA)
--------------------------------------------------

Shape of dataset : (150, 5)

Missing values per column :
...

Class distribution (species count)
...

--------------------------------------------------
Built the model
--------------------------------------------------

Model gets created successfully

--------------------------------------------------
Train the model
--------------------------------------------------

Model trained successfully

--------------------------------------------------
Evaluate the model performance
--------------------------------------------------

Accuracy of model is : ...
```

It also displays the **scatter plot**, **confusion matrix**, and **classification report**.

---

## 📌 Functions Used

| Function | Purpose |
|---|---|
| `DataLoading()` | Loads the dataset |
| `DataAnalysis()` | Performs EDA |
| `DataSeperate()` | Separates X and Y |
| `DataVisualization()` | Creates visualization |
| `DataSplitting()` | Splits data into training/testing sets |
| `CreateModel()` | Creates Decision Tree model |
| `TrainModel()` | Trains the model |
| `TestModel()` | Generates predictions |
| `EvaluateModel()` | Evaluates model performance |
| `main()` | Controls the complete workflow |

---

## 🧠 Key Concepts Demonstrated

- Dataset loading
- Exploratory Data Analysis
- DataFrame operations
- Feature selection
- Target variable selection
- Data visualization
- Train-test split
- Supervised learning
- Classification
- Decision Tree
- Model training
- Model prediction
- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score
- Classification report

---

## 🚀 Future Improvements

The project can be extended by:

- Using `test_size=0.2` for an 80:20 train-test split
- Adding a heatmap for the confusion matrix
- Performing hyperparameter tuning
- Adding cross-validation
- Comparing multiple classification algorithms
- Saving the trained model using `joblib`
- Creating a user-input prediction system
- Adding a graphical user interface

---

## 👨‍💻 Author

**Sahil Ashok Dhole**

M.Sc. Computer Science Student

---

## 📜 License

This project is created for **educational and academic purposes**.
