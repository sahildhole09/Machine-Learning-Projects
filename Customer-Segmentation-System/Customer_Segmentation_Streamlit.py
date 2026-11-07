import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")
st.write("Using K-Means Clustering Algorithm")

df = pd.read_csv("Mall_Customers.csv")

df["AnnualIncome"] = df["AnnualIncome"].fillna(df["AnnualIncome"].median())
df["SpendingScore"] = df["SpendingScore"].fillna(df["SpendingScore"].median())

st.sidebar.header("Model Settings")

k = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)

st.subheader("📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("Total Clusters", k)
col3.metric("Average Income", round(df["AnnualIncome"].mean(), 2))
col4.metric("Average Spending", round(df["SpendingScore"].mean(), 2))

X = df[["AnnualIncome", "SpendingScore"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("📈 Elbow Method")

wcss = []

for i in range(1, 11):
    model = KMeans(n_clusters=i, 
                   random_state=42, 
                   n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

fig1, ax1 = plt.subplots(figsize=(8,3))

ax1.plot(range(1, 11), wcss, marker="o")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")
ax1.grid()

st.pyplot(fig1)

st.subheader("🎯 Customer Segmentation")

model = KMeans(n_clusters=k, 
               random_state=42, 
               n_init=10)

df["Cluster"] = model.fit_predict(X_scaled)

fig2, ax2 = plt.subplots(figsize=(8,3))

for i in range(k):
    data = df[df["Cluster"] == i]

    ax2.scatter(
        data["AnnualIncome"],
        data["SpendingScore"],
        label=f"Cluster {i}"
    )

ax2.set_xlabel("Annual Income")
ax2.set_ylabel("Spending Score")
ax2.set_title("Customer Groups")
ax2.legend()
ax2.grid()

st.pyplot(fig2)

st.success("Customer Segmentation Completed Successfully!")

st.subheader("📋 Cluster Analysis")

summary = df.groupby("Cluster").agg({
    "CustomerID": "count",
    "AnnualIncome": "mean",
    "SpendingScore": "mean"
})

summary.columns = [
    "Total Customers",
    "Average Income",
    "Average Spending"
]

st.dataframe(summary.round(2))

st.subheader("👥 Customer Details")

selected = st.selectbox(
    "Select Customer Group",
    ["All Groups"] + list(range(k))
)

if selected == "All Groups":
    result = df
else:
    result = df[df["Cluster"] == selected]

st.dataframe(result, use_container_width=True)

st.subheader("⬇️ Download Results")

csv = df.to_csv(index=False)

st.download_button(
    "Download Customer Data",
    csv,
    "Customer_Segmentation.csv",
    "text/csv"
)