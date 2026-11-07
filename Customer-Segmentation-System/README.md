# Customer Segmentation System

A customer segmentation project that uses **K-Means Clustering** to group mall customers based on their **Annual Income** and **Spending Score**. The project includes a Python machine-learning workflow and an interactive Streamlit dashboard for exploring customer groups.

## Project Overview

Customer segmentation helps businesses understand customers with similar purchasing patterns. This project:

- Loads customer information from a CSV dataset.
- Checks dataset dimensions, columns, missing values, and descriptive statistics.
- Replaces missing values in `Age`, `AnnualIncome`, and `SpendingScore` with the corresponding median.
- Selects `AnnualIncome` and `SpendingScore` as clustering features.
- Standardizes the selected features using `StandardScaler`.
- Uses the Elbow Method to calculate and plot WCSS for cluster counts from 1 through 10.
- Applies K-Means to assign each customer to a cluster.
- Displays cluster summaries and customer-level results in Streamlit.
- Allows the user to choose the number of clusters and download the clustered customer data as CSV.

## Project Files

| File | Description |
|---|---|
| `Customer_Kmeans.py` | Main machine-learning workflow: data loading, analysis, feature selection, scaling, Elbow Method, visualization, and final K-Means clustering. |
| `Customer_Segmentation_Streamlit.py` | Interactive dashboard with metrics, Elbow chart, customer-cluster scatter plot, cluster summary, customer filtering, and CSV download. |
| `Mall_Customers.csv` | Customer dataset used by the project. |
| `requirements.txt` | Python package requirements. |

## Dataset

The application reads the dataset from `Mall_Customers.csv`.

The workflow expects these columns:

| Column | Description |
|---|---|
| `CustomerID` | Customer identifier. |
| `Age` | Customer age. Used for missing-value handling and displayed in the customer data. |
| `AnnualIncome` | Annual income. Used as a clustering feature. |
| `SpendingScore` | Spending score. Used as a clustering feature. |

**Clustering features:** `AnnualIncome` and `SpendingScore`.

The CSV must be placed in the project folder alongside the Python files. The scripts use the relative filename `Mall_Customers.csv`.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Installation

### 1. Clone or download the project

```bash
git clone <your-repository-url>
cd <project-folder>
```

Or download the project files and place them in one folder.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The supplied requirements file contains many packages beyond the imports used directly by these two scripts. If installation is unnecessarily large or encounters platform-specific package issues, create a smaller project-specific requirements file containing the packages directly used by the application:

```text
pandas
matplotlib
scikit-learn
streamlit
```

## Run the Machine-Learning Workflow

Make sure `Mall_Customers.csv` is in the same directory, then run:

```bash
python Customer_Kmeans.py
```

The script runs these stages:

1. **LoadData** – reads the CSV and displays the first five rows.
2. **AnalyseData** – reports dataset shape, columns, missing values, and descriptive statistics; fills missing `Age`, `AnnualIncome`, and `SpendingScore` values with their medians.
3. **FeatureSelection** – selects `AnnualIncome` and `SpendingScore`.
4. **FeatureScaling** – standardizes the selected features using `StandardScaler`.
5. **ElbowMethod** – fits K-Means for `k = 1` to `10` and records WCSS (inertia).
6. **DataVisualization** – displays the Elbow Method chart.
7. **FinalModel** – fits K-Means with `4` clusters and adds the predicted `Cluster` column to the dataframe.

The final model in the standalone script uses `n_clusters=4`, `random_state=42`, and `n_init=10`.

## Run the Streamlit Dashboard

Start the dashboard from the project directory:

```bash
streamlit run Customer_Segmentation_Streamlit.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser.

### Dashboard Features

- **Overview metrics:** total customers, selected number of clusters, average annual income, and average spending score.
- **Cluster count control:** sidebar slider lets you select between 2 and 10 clusters (default 4).
- **Elbow Method chart:** shows WCSS for cluster counts 1–10.
- **Customer segmentation chart:** scatter plot of annual income versus spending score, colored by cluster.
- **Cluster analysis table:** customer count and average income and spending score for each cluster.
- **Customer details:** view all customers or filter by a selected cluster.
- **CSV export:** download the dataset with its assigned cluster labels.

## How K-Means Works in This Project

1. The model uses each customer's annual income and spending score as a two-dimensional feature vector.
2. `StandardScaler` puts the two features on a comparable scale before clustering.
3. K-Means identifies groups of customers with similar feature values.
4. The Elbow Method calculates WCSS for different values of `k`; the chart can help identify a suitable cluster count.
5. The dashboard fits the final model using the cluster count selected in the sidebar and assigns a cluster label to every row.

Cluster labels (`0`, `1`, `2`, and so on) are identifiers. A higher label does **not** inherently mean a better or worse customer segment.

## Output

The standalone script displays the dataset and Elbow chart, then prints the dataframe with a `Cluster` column.

The Streamlit dashboard displays charts and tables and provides a downloadable file named:

```text
Customer_Segmentation.csv
```

The downloaded CSV includes the original customer columns and the assigned `Cluster` label.

## Notes

- The dashboard and standalone script both expect the CSV filename `Mall_Customers.csv`.
- The dashboard fills missing values for `AnnualIncome` and `SpendingScore`. The standalone workflow also fills missing `Age`.
- The standalone workflow uses four clusters. The dashboard allows the user to select from 2 to 10 clusters.
- The Elbow Method is a visual aid; the best cluster count should be selected by interpreting the chart and the project's needs.
- This project is an exploratory segmentation example. Cluster labels describe patterns in the supplied dataset and should not be treated as customer value ratings.

## Author

**Sahil Ashok Dhole**
