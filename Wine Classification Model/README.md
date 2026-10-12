# Wine Predictor

A machine learning project that uses a **Decision Tree Classifier** to predict the wine class from chemical measurements. The script loads a CSV dataset, performs basic exploratory data analysis, separates features and labels, trains and tests a classifier, and reports evaluation metrics.

## Features

- Loads wine data from `WinePredictor.csv`
- Displays the first five dataset rows
- Prints dataset dimensions, column names, missing-value counts, and class distribution
- Uses 13 wine-related measurements as input features
- Splits the data into training and testing sets (70% / 30%)
- Trains a `DecisionTreeClassifier` with `random_state=42`
- Predicts classes for the test set
- Reports accuracy, confusion matrix, and classification report

## Project Structure

```text
WinePredictor/
├── WinePredictor.py
├── WinePredictor.csv
└── README.md
```

> Keep `WinePredictor.csv` in the same working directory from which you run the script. The script currently uses this exact filename.

## Requirements

- Python 3.8+ (or another Python version compatible with the installed dependencies)
- pandas
- scikit-learn

## Installation

1. Install Python if it is not already installed.
2. (Optional) Create and activate a virtual environment.

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required libraries:

   ```bash
   pip install pandas scikit-learn
   ```

## Dataset

The script expects a file named `WinePredictor.csv` containing the following columns:

- `Alcohol`
- `Malic acid`
- `Ash`
- `Alcalinity of ash`
- `Magnesium`
- `Total phenols`
- `Flavanoids`
- `Nonflavanoid phenols`
- `Proanthocyanins`
- `Color intensity`
- `Hue`
- `OD280/OD315 of diluted wines`
- `Proline`
- `Class` — the target label to predict

The 13 measurement columns are the independent variables (`X`), and `Class` is the dependent variable (`Y`). The CSV must include these column names exactly as written in the script.

## How to Run

Place `WinePredictor.py` and `WinePredictor.csv` in the same directory, then run:

```bash
python WinePredictor.py
```

The program runs the following pipeline:

1. **Load data** — reads the CSV and prints the first five rows.
2. **Analyse data** — displays shape, columns, missing values, class counts, and a statistical summary.
3. **Separate features and target** — selects the 13 measurement columns and the `Class` target.
4. **Split data** — uses `train_test_split(test_size=0.3, random_state=42)`.
5. **Create model** — initializes `DecisionTreeClassifier(random_state=42)`.
6. **Train model** — fits the classifier on the training data.
7. **Test model** — predicts classes for the held-out test data.
8. **Evaluate model** — prints accuracy, confusion matrix, and classification report.

## Evaluation Metrics

- **Accuracy:** proportion of test examples predicted correctly, displayed as a percentage.
- **Confusion matrix:** compares actual and predicted classes.
- **Classification report:** provides precision, recall, F1-score, and support for each class.

The actual metrics depend on the contents of `WinePredictor.csv` and are printed when you run the script.

## Notes

- The CSV must contain all required feature columns and the `Class` target column.
- The script does not include a saved model or a separate prediction interface; it trains and evaluates the model each time it runs.
- Ensure the dataset is suitable for classification and has valid values for the selected columns.
- The script uses `df.describe` without calling it as `df.describe()`, so the current output prints the method representation rather than the statistical summary. To display the summary statistics, change that line to `df.describe()`.
- `train_test_split` does not specify `stratify=Y`, so the class proportions in the train and test sets are not explicitly preserved.

## Author

**Sahil Ashok Dhole**
