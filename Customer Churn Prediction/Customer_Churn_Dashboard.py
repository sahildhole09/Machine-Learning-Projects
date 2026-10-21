import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Customer Churn Prediction Dashboard")
st.subheader("👨‍💻 By Sahil Ashok Dhole : Student of Marvellous Infosystems")
st.caption("Interactive Machine Learning Application for Customer Retention Analysis")

st.divider()

model = joblib.load("Customer_Churn.pkl")

col_input, col_result = st.columns([1, 1], gap="medium")

with col_input:
    with st.container(border=True):
        st.subheader("📋 Enter Customer Details")
        st.write("Enter the customer parameters below :")

        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            age = st.number_input("🎂 Customer Age", min_value=18, max_value=100, value=30, step=1)
            tenure = st.number_input("⏳ Account Tenure (Months)", min_value=0, max_value=120, value=12, step=1)

        with sub_col2:
            charges = st.number_input("💵 Monthly Charges ($)", min_value=10.0, max_value=500.0, value=65.0, step=1.0)
            calls = st.number_input("📞 Support Calls Made", min_value=0, max_value=50, value=2, step=1)

        st.write("")
        predict_btn = st.button("🚀 Predict Customer Churn", type="primary",use_container_width=True)

with col_result:
    with st.container(border=True):
        st.subheader("🎯 Prediction Result & Risk Assessment")

        if predict_btn:
            new_customer = pd.DataFrame({
                "Age": [age],
                "Tenure": [tenure],
                "MonthlyCharges": [charges],
                "SupportCalls": [calls]
            })

            prediction = model.predict(new_customer)[0]
            probability = model.predict_proba(new_customer)[0][1] * 100

            m1, m2 = st.columns(2)

            m1.metric(
                label="Churn Probability",
                value=f"{probability:.1f}%",
                delta="High Risk" if prediction == 1 else "Safe",
                delta_color="inverse"
            )
            m2.metric(
                label="Predicted Status",
                value="Likely to Churn" if prediction == 1 else "Likely to Stay"
            )

            st.write("📊 **Risk Meter :**")
            st.progress(int(min(max(probability, 0), 100)))

            if prediction == 1:
                st.error("🚨 **High Churn Risk!** Customer is likely to leave the service.")
                st.warning("💡 **Retention Action:** High risk detected. Consider offering a loyalty discount or dedicated support outreach.")
            else:
                st.success("🎉 **Safe Customer!** Customer is satisfied and likely to stay.")
                st.balloons()
                st.info("💡 **Retention Action:** Customer health is strong. Good candidate for loyalty rewards or long-term plan upgrades.")
        else:
            st.info("👈 Enter customer details on the left and click **'Predict Customer Churn'** to view the live risk assessment.")

st.divider()
st.caption("Customer Churn Prediction System | Marvellous Infosystems - Sahil Ashok Dhole")
