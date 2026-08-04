import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below to predict the loan status.")

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

education = st.selectbox(
    "Education",
    label_encoders["education"].classes_
)

self_employed = st.selectbox(
    "Self Employed",
    label_encoders["self_employed"].classes_
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=12
)

cibil = st.slider(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=750
)

residential = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=500000
)

commercial = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=0
)

luxury = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=0
)

bank = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=500000
)

if st.button("Predict Loan Status"):

    education_encoded = label_encoders["education"].transform([education])[0]
    self_encoded = label_encoders["self_employed"].transform([self_employed])[0]

    input_data = pd.DataFrame({
        "no_of_dependents": [dependents],
        "education": [education_encoded],
        "self_employed": [self_encoded],
        "income_annum": [income],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil],
        "residential_assets_value": [residential],
        "commercial_assets_value": [commercial],
        "luxury_assets_value": [luxury],
        "bank_asset_value": [bank]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    loan_status = label_encoders["loan_status"].inverse_transform([prediction])[0]

    st.subheader("Prediction Result")

    if loan_status.lower() == "approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Confidence:** {max(probability) * 100:.2f}%")

    st.progress(float(max(probability)))