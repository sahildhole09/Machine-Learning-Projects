# ⚡ Customer Churn Prediction System

An end-to-end Machine Learning and Decision Intelligence application designed to predict customer churn risk, analyze customer retention drivers, and deliver actionable business recommendations through an interactive **Streamlit** dashboard.

---

## 👨‍💻 Project Information & Credits

- **Project Title:** Customer Churn Prediction & Retention Analytics
- **Author:** **Sahil Ashok Dhole**
- **Institution:** Student of **Marvellous Infosystems**
- **Domain:** Telecommunications / SaaS Subscription Analytics
- **Machine Learning Algorithm:** Logistic Regression (`L2` regularization)
- **Deployment Platform:** Streamlit Web Application

---

## 📌 Problem Statement

Customer churn represents the rate at which customers discontinue their subscriptions or service contracts. Acquiring new customers typically costs **5× to 7× more** than retaining existing ones. 

This project aims to:
1. Identify high-risk customers early based on behavioral and billing metrics.
2. Quantify the probability of churn (0% to 100%).
3. Deliver clear, automated business interventions (e.g., priority support escalation or loyalty discounts) to prevent customer loss before it occurs.

---

## 📊 Dataset Overview

The dataset (`customer_churn.csv`) contains **600 customer records** with 4 independent behavioral features and 1 binary target variable.

| Feature Name | Data Type | Description | Value Range |
| :--- | :---: | :--- | :---: |
| `Age` | Integer | Customer age in years | 18 – 70 |
| `Tenure` | Integer | Duration as an active subscriber (in months) | 1 – 72 |
| `MonthlyCharges` | Numeric | Recurring monthly billing charge ($) | $25 – $113 |
| `SupportCalls` | Integer | Number of customer support / complaint calls | 0 – 9 |
| **`Churn`** | Binary | **Target Variable** (0 = Retained, 1 = Churned) | 0 or 1 |

### Target Distribution
- **Retained (0):** 365 customers (~60.8%)
- **Churned (1):** 235 customers (~39.2%)

---

## 🏗️ Project Architecture & Workflow

The project follows a clean, modular architectural pipeline implemented across dedicated files:

```
├── customer_churn.csv             # Raw dataset (600 records)
├── Customer_ChurnLR.py            # Complete ML training & evaluation pipeline
├── Customer_Churn.pkl             # Serialized trained Logistic Regression model
├── Customer_Churn_Dashboard.py    # Streamlit wide-screen interactive web dashboard
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

### Machine Learning Pipeline (`Customer_ChurnLR.py`)

The training script is structured into modular, reusable functions:

```
[ DataLoading ]
       ↓
[ DataAnalysis (EDA) ]
       ↓
[ DataSeperate (X & Y) ]
       ↓
[ DataSplitting (50% Train / 50% Test) ]
       ↓
[ CreateModel (LogisticRegression) ]
       ↓
[ TrainModel (fit) ]
       ↓
[ TestModel (predict) ]
       ↓
[ EvaluateModel (Accuracy, Confusion Matrix, Classification Report) ]
       ↓
[ PreserveModel (Customer_Churn.pkl) ]
```

---

## 📈 Model Performance & Evaluation

The Logistic Regression model was trained on 300 records and evaluated on an independent test split of 300 records (`random_state=42`, 50% split).

### Key Metrics

| Metric | Score |
| :--- | :---: |
| **Overall Accuracy** | **99.0%** |
| **Macro Average F1-Score** | **0.99** |
| **Weighted Average F1-Score** | **0.99** |

### Detailed Classification Report

```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       185
           1       0.99      0.98      0.99       115

    accuracy                           0.99       300
   macro avg       0.99      0.99      0.99       300
weighted avg       0.99      0.99      0.99       300
```

### Confusion Matrix

```text
                 Predicted Retained (0)   Predicted Churned (1)
Actual Retained (0)       184                       1
Actual Churned (1)          2                     113
```

- **True Negatives (TN):** 184 (Correctly predicted retained)
- **True Positives (TP):** 113 (Correctly predicted churners)
- **False Positives (FP):** 1
- **False Negatives (FN):** 2

---

## 💡 Key Business Findings & Feature Insights

Analysis of the model's learned weights reveals key operational drivers of churn:

1. **Support Calls (`+` Impact):** The strongest churn accelerator. Customers with $\ge 4$ support calls show an exponential spike in churn probability due to unresolved service frustration.
2. **Monthly Charges (`+` Impact):** High recurring fees ($\ge \$70$) significantly increase sensitivity and attrition risk.
3. **Account Tenure (`-` Impact):** Customers who have stayed beyond 12–18 months develop loyalty and display dramatically lower churn rates.
4. **Customer Age (`-` Impact):** Mature customer demographics exhibit more stable subscription retention than younger cohorts.

---

## 🚀 Streamlit Web Dashboard Features

The dashboard (`Customer_Churn_Dashboard.py`) provides an intuitive, executive-ready interface:

- **Full Screen UI Space (`layout="wide"`):** Spans the entire browser display using side-by-side bordered containers (`st.container(border=True)`).
- **Precise Number Inputs (`st.number_input`):** Stepper inputs for `Age`, `Tenure`, `MonthlyCharges`, and `SupportCalls` with practical default values.
- **Real-Time Probability Meter:** Calculates exact churn probability (0.0% to 100.0%) and renders an animated risk meter progress bar (`st.progress`).
- **Dynamic KPI Metrics Cards:** Highlights current risk score and categorical outcome (`Likely to Churn` vs `Likely to Stay`).
- **Prescriptive Retention Actions:** Automatically suggests operational countermeasures:
  - *High Risk:* Assign priority support account manager or offer a loyalty discount.
  - *Low Risk:* Identify candidate for loyalty rewards or long-term plan upgrades.
- **Interactive Celebration:** Triggers an animated balloon celebration (`st.balloons()`) when a customer is identified as retained.

---

## 🛠️ Installation & Setup Guide

### 1. Prerequisites
Make sure **Python 3.10+** is installed on your operating system.

### 2. Clone or Extract the Project
Open terminal / command prompt and navigate to the project directory:
```bash
cd "Customer Churn Prediction"
```

### 3. (Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install the required packages using pip:
```bash
pip install pandas numpy scikit-learn joblib streamlit
```

*(Alternatively, install from `requirements.txt`: `pip install -r requirements.txt`)*

---

## 🖥️ Running the Application

### Step 1: Train & Save the Machine Learning Model
To run the end-to-end training pipeline and regenerate `Customer_Churn.pkl`:
```bash
python Customer_ChurnLR.py
```
*This will print dataset statistics, train the Logistic Regression model, evaluate accuracy (99.0%), and save the serialized model file.*

### Step 2: Launch the Streamlit Interactive Dashboard
Start the web dashboard by running:
```bash
streamlit run Customer_Churn_Dashboard.py
```

Streamlit will launch local and network URLs (typically `http://localhost:8501`). Open this link in your web browser to interact with the application.

---

## 📂 Project Structure

```plaintext
Customer Churn Prediction/
│
├── customer_churn.csv             # Dataset with 600 customer profiles
├── Customer_ChurnLR.py            # Core ML training and evaluation script
├── Customer_Churn.pkl             # Saved Scikit-Learn Logistic Regression model
├── Customer_Churn_Dashboard.py    # Streamlit web application
├── requirements.txt               # Dependencies list
└── README.md                      # Comprehensive project documentation
```

---

## 🔮 Future Enhancements

- [ ] Incorporate additional customer features (contract type, internet service tier, payment method).
- [ ] Experiment with ensemble classifiers (Random Forest, XGBoost) and hyperparameter tuning.
- [ ] Add batch customer CSV upload and report download capabilities.
- [ ] Deploy the dashboard to Streamlit Community Cloud or Docker container.

---

## 📜 License & Acknowledgments

- **Developed by:** **Sahil Ashok Dhole**
- **Special Thanks:** **Marvellous Infosystems** for guidance, structured machine learning curriculum, and mentorship.
- **License:** Open for academic and portfolio demonstration purposes.

