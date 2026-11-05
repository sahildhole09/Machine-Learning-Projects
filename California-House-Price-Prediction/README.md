# California House Price Prediction

A machine learning project that predicts California house prices using
two regression algorithms: **Decision Tree Regressor** and **Linear
Regression**. It also provides an interactive **Streamlit** interface
for exploring the dataset, visualizing relationships, evaluating models,
and predicting a house price from user-entered details.

## Project Features

-   Load the California housing dataset from a CSV file.
-   Perform exploratory data analysis (EDA), including dataset shape,
    column names, missing values, and descriptive statistics.
-   Train and test a Decision Tree regression model.
-   Train and test a Linear Regression model.
-   Evaluate regression models using Mean Squared Error (MSE), Root Mean
    Squared Error (RMSE), and R² score (where implemented).
-   Visualize the data and model results in Streamlit:
    -   House price (`target`) distribution
    -   Feature correlation heatmap
    -   Median income versus house price
    -   Actual versus predicted house prices
    -   Decision Tree feature importance
-   Enter house characteristics in the Streamlit app and get a predicted
    price.

## Technologies Used

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Streamlit
-   Joblib

## Project Files

  ---------------------------------------------------------------------------
  File                                    Description
  --------------------------------------- -----------------------------------
  `CalHousePricePredictionDT.py`          Command-line workflow using
                                          `DecisionTreeRegressor`. Trains,
                                          evaluates, predicts a sample house
                                          price, and saves the trained model.

  `CalHousePricePredictionLR.py`          Command-line workflow using
                                          `LinearRegression`. Trains,
                                          evaluates, predicts a sample house
                                          price, and saves the trained model.

  `CalHousePricePredictionStreamlit.py`   Streamlit application with
                                          navigation for Home, Data Analysis,
                                          Data Visualization, Model Training,
                                          and New Prediction.

  `california_housing.csv`                Dataset used by the scripts and
                                          Streamlit application.

  `requirement.txt`                       Python package requirements
                                          supplied with the project.
  ---------------------------------------------------------------------------

## Dataset

The project expects a CSV file named **`california_housing.csv`** in the
project directory.

The code uses these eight input features:

-   `MedInc` -- median income
-   `HouseAge` -- house age
-   `AveRooms` -- average rooms
-   `AveBedrms` -- average bedrooms
-   `Population` -- population
-   `AveOccup` -- average occupancy
-   `Latitude` -- latitude
-   `Longitude` -- longitude

The column **`target`** is used as the value to predict.

## Installation

1.  Clone or download this project.

2.  Open a terminal in the project directory.

3.  (Recommended) Create and activate a virtual environment:

    ``` bash
    python -m venv venv
    ```

    Windows:

    ``` bash
    venv\Scripts\activate
    ```

    macOS/Linux:

    ``` bash
    source venv/bin/activate
    ```

4.  Install the packages listed in the supplied requirements file:

    ``` bash
    pip install -r requirement.txt
    ```

If you maintain a smaller dependency file instead, the application
imports `streamlit`, `numpy`, `pandas`, `matplotlib`, `seaborn`, and
`scikit-learn`; the standalone scripts also use `joblib`.

## Run the Streamlit Application

Make sure `CalHousePricePredictionStreamlit.py` and
`california_housing.csv` are in the same directory, then run:

``` bash
streamlit run CalHousePricePredictionStreamlit.py
```

Streamlit will display a local URL in the terminal. Open that URL in
your browser.

### Streamlit Navigation

Use the sidebar to select an operation:

1.  **Home** -- displays dataset summary metrics and a preview of the
    first rows.
2.  **Data Analysis** -- displays dataset shape, column names,
    missing-value counts, and descriptive statistics.
3.  **Data Visualization** -- displays the target distribution,
    correlation heatmap, income-versus-target scatter plot,
    actual-versus-predicted plot, and (for Decision Tree) feature
    importance.
4.  **Model Training** -- displays MSE, RMSE, R² score, and a sample of
    actual and predicted values.
5.  **New Prediction** -- enter the eight feature values and click
    **Predict House Price**.

Choose either **Decision Tree** or **Linear Regression** from the
sidebar's ML algorithm selector.

## Run the Standalone Python Models

Run either script from the project directory:

``` bash
python CalHousePricePredictionDT.py
```

``` bash
python CalHousePricePredictionLR.py
```

The scripts split the data with `test_size=0.5` and `random_state=42`,
train the selected regressor, print evaluation results, and make a
prediction using the sample input values defined in each script.

The trained models are saved using Joblib:

-   Decision Tree: `CalHousePricePredModel.pkl`
-   Linear Regression: `CalHousePricePredLRModel.pkl`

## Model Evaluation

The project uses regression metrics:

-   **MSE (Mean Squared Error):** average squared difference between
    actual and predicted values; lower is better.
-   **RMSE (Root Mean Squared Error):** square root of MSE, expressed in
    the target's units; lower is better.
-   **R² score:** indicates how well the model explains variation in the
    target; a higher value is generally better.

The Streamlit interface calculates and displays all three metrics for
the selected model. The Decision Tree standalone script reports MSE and
R²; the Linear Regression standalone script reports MSE, RMSE, and R².

## Notes

-   Keep the CSV filename exactly as `california_housing.csv`, because
    that is the filename referenced in the Python files.
-   The Streamlit app displays the predicted price as
    `model prediction × 100,000` and labels the target distribution in
    units of \$100,000. The app's displayed dollar value is therefore a
    scaled presentation of the model output.
-   Feature-importance visualization is available only when Decision
    Tree is selected; the app displays an information message for Linear
    Regression.
-   The scripts and Streamlit app use a 50% test split and a fixed
    random state of 42.

## Author

**Sahil Ashok Dhole**
