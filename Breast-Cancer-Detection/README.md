# Breast Cancer Detection Using Decision Tree

A machine learning project that uses a **Decision Tree Classifier** to
classify breast cancer data from a CSV dataset. The Python script loads
and explores the dataset, separates features and target labels, splits
the data into training and testing sets, scales the features, trains a
model, and reports evaluation metrics.

## Project Structure

``` text
Breast-Cancer-Detection/
├── BreastCancerDetectionDT.py
├── breast_cancer.csv
├── README.md
└── requirements.txt
```

## Features

-   Loads the dataset from `breast_cancer.csv` using Pandas.
-   Displays the dataset shape, first few rows, column names,
    missing-value counts, target distribution, and descriptive
    statistics.
-   Separates independent variables (`X`) from the dependent variable
    (`target`).
-   Splits the data into training and testing sets using a 70/30 split.
-   Scales feature values with `StandardScaler`.
-   Trains a `DecisionTreeClassifier` with `max_depth=5` and
    `random_state=42`.
-   Evaluates predictions using accuracy, a confusion matrix, and a
    classification report.

## Requirements

-   Python 3.8 or newer recommended
-   pandas
-   scikit-learn

## Installation

1.  Install Python if it is not already installed.

2.  Place `BreastCancerDetectionDT.py` and `breast_cancer.csv` in the
    same directory.

3.  Open a terminal in that directory.

4.  Install the required packages:

    ``` bash
    pip install pandas scikit-learn
    ```

## Dataset

The script expects a file named **`breast_cancer.csv`** in the current
working directory. It expects the dataset to contain a column named
**`target`**. All other columns are used as input features.

The code prints the dataset dimensions, sample records, missing values,
target-class counts, and summary statistics before training.

## How to Run

Run the script from the directory containing both files:

``` bash
python BreastCancerDetectionDT.py
```

The program runs the following workflow:

1.  Load the CSV dataset.
2.  Perform basic exploratory data analysis.
3.  Split the dataset into features (`X`) and target labels (`Y`).
4.  Split the data into training and testing sets (`test_size=0.3`,
    `random_state=42`).
5.  Standardize the feature values.
6.  Create the Decision Tree model.
7.  Train the model.
8.  Predict labels for the test set.
9.  Print the evaluation results.

## Model Configuration

  Setting                   Value
  ------------------------- --------------------------
  Algorithm                 Decision Tree Classifier
  Maximum tree depth        `5`
  Random state              `42`
  Test set proportion       `30%`
  Training set proportion   `70%`
  Feature scaler            `StandardScaler`

## Evaluation Metrics

The script reports:

-   **Accuracy:** the proportion of test examples classified correctly.
-   **Confusion matrix:** compares actual and predicted class labels.
-   **Classification report:** displays precision, recall, F1-score, and
    support for each class.

The actual results depend on the dataset and the execution environment;
no fixed accuracy is claimed here.

## Important Implementation Note

In the current script, `StandardScaler` is fitted separately to both the
training and test data. For a more reliable evaluation, fit the scaler
only on the training data and use that fitted scaler to transform the
test data:

``` python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

This avoids using test-set statistics during preprocessing. Also,
because this is a medical-related classification project, its
predictions are for educational purposes only and must **not** be used
as a substitute for professional medical diagnosis or advice.

## Author

**Sahil Ashok Dhole**

## License

No license is specified in the provided project files. Add a `LICENSE`
file if you intend to distribute the project with a particular license.
