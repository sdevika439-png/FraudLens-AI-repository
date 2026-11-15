
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------
# FRAUDLENS AI - UPI FRAUD DETECTION
# ---------------------------------------

MODEL_FILE = "upi_fraud_model.joblib"

FEATURES = [
    "amount",
    "hour",
    "transactions_last_1h",
    "new_payee",
    "device_changed",
    "location_changed",
    "failed_upi_attempts",
    "beneficiary_age_days",
    "amount_vs_avg_ratio"
]

st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="wide"
)


# ---------------------------------------
# LOAD MACHINE LEARNING MODEL
# ---------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)


model = load_model()


# ---------------------------------------
# PAGE DESIGN
# ---------------------------------------

st.title("🔍 FraudLens AI")
st.subheader("UPI & Digital Payment Fraud Detection")

st.write(
    "An academic machine learning prototype "
    "for analysing suspicious UPI-like transactions."
)

st.warning(
    "Educational prototype only. Scores are experimental "
    "and are not validated real-world fraud probabilities."
)

if model is None:
    st.error(
        "Trained model not found. Please run "
        "'python3 train_upi_model.py' before launching the dashboard."
    )
    st.stop()


# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

def prepare_data(data):

    data = data.copy()

    yes_no_columns = [
        "new_payee",
        "device_changed",
        "location_changed"
    ]

    for column in yes_no_columns:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "yes": 1,
                    "no": 0,
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0
                })
            )

    for column in FEATURES:
        if column not in data.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data[FEATURES].isna().all(axis=1).any():
        raise ValueError(
            "Some transactions have no usable feature values."
        )

    return data


def risk_category(score):

    if score >= 60:
        return "High Risk"

    elif score >= 30:
        return "Medium Risk"

    return "Low Risk"


def investigation_flags(row):

    flags = []

    if row["amount"] > 15000:
        flags.append("High transaction amount")

    if row["hour"] <= 5:
        flags.append("Transaction during unusual hours")

    if row["transactions_last_1h"] >= 5:
        flags.append("High transaction frequency")

    if row["new_payee"] == 1:
        flags.append("New beneficiary")

    if row["device_changed"] == 1:
        flags.append("New or changed device")

    if row["location_changed"] == 1:
        flags.append("Changed transaction location")

    if row["failed_upi_attempts"] >= 2:
        flags.append("Multiple failed UPI attempts")

    if row["beneficiary_age_days"] <= 2:
        flags.append("Recently added beneficiary")

    if row["amount_vs_avg_ratio"] >= 4:
        flags.append("Amount much higher than usual")

    if not flags:
        flags.append("No predefined warning indicators detected")

    return ", ".join(flags)


def analyse_transactions(data):

    prepared = prepare_data(data)

    X = prepared[FEATURES]

    probabilities = model.predict_proba(X)[:, 1]

    results = data.copy()

    results["risk_score"] = np.round(
        probabilities * 100, 2
    )

    results["risk_category"] = [
        risk_category(score)
        for score in results["risk_score"]
    ]

    results["investigation_flags"] = [
        investigation_flags(row)
        for _, row in prepared.iterrows()
    ]

    return results


# ---------------------------------------
# DASHBOARD TABS
# ---------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Single UPI Analysis",
    "Bulk CSV Analysis",
    "Model Information"
])


# ---------------------------------------
# TAB 1: SINGLE TRANSACTION ANALYSIS
# ---------------------------------------

with tab1:

    st.header("Analyse a UPI Transaction")

    with st.form("upi_form"):

        col1, col2 = st.columns(2)

        with col1:

            amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=1.0,
                value=1500.0
            )

            hour = st.slider(
                "Transaction Hour",
                min_value=0,
                max_value=23,
                value=14
            )

            transactions_last_1h = st.number_input(
                "Transactions in Last 1 Hour",
                min_value=0,
                value=1
            )

            failed_upi_attempts = st.number_input(
                "Failed UPI Attempts",
                min_value=0,
                value=0
            )

            beneficiary_age_days = st.number_input(
                "Beneficiary Age (Days)",
                min_value=0,
                value=100
            )

        with col2:

            new_payee = st.selectbox(
                "Is this a New Beneficiary?",
                ["No", "Yes"]
            )

            device_changed = st.selectbox(
                "Has the Device Changed?",
                ["No", "Yes"]
            )

            location_changed = st.selectbox(
                "Has the Location Changed?",
                ["No", "Yes"]
            )

            amount_vs_avg_ratio = st.number_input(
                "Amount Compared to Usual Transaction Amount",
                min_value=0.1,
                value=1.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Analyse Transaction",
            use_container_width=True
        )

    if submitted:

        transaction = pd.DataFrame([{
            "amount": amount,
            "hour": hour,
            "transactions_last_1h": transactions_last_1h,
            "new_payee": new_payee,
            "device_changed": device_changed,
            "location_changed": location_changed,
            "failed_upi_attempts": failed_upi_attempts,
            "beneficiary_age_days": beneficiary_age_days,
            "amount_vs_avg_ratio": amount_vs_avg_ratio
        }])

        result = analyse_transactions(transaction)

        score = float(result["risk_score"].iloc[0])
        category = result["risk_category"].iloc[0]
        flags = result["investigation_flags"].iloc[0]

        st.divider()

        st.subheader("Transaction Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Experimental Risk Score", f"{score:.2f}%")

        with col2:
            st.metric("Risk Category", category)

        st.progress(int(round(score)))

        st.write("**Investigation Indicators**")

        for flag in flags.split(", "):
            st.write("•", flag)

        st.info(
            "This result is an experimental model output. "
            "It is not proof of fraud and should not be used "
            "to automatically block payments."
        )


# ---------------------------------------
# TAB 2: BULK CSV ANALYSIS
# ---------------------------------------

with tab2:

    st.header("Bulk UPI Transaction Analysis")

    st.write(
        "Upload a CSV file containing UPI transaction features."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Transaction Data")

            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

            if st.button("Analyse All Transactions"):

                results = analyse_transactions(uploaded_data)

                st.subheader("Analysis Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                high_risk = (
                    results["risk_category"] == "High Risk"
                ).sum()

                medium_risk = (
                    results["risk_category"] == "Medium Risk"
                ).sum()

                low_risk = (
                    results["risk_category"] == "Low Risk"
                ).sum()

                col1, col2, col3 = st.columns(3)

                col1.metric("High Risk", int(high_risk))
                col2.metric("Medium Risk", int(medium_risk))
                col3.metric("Low Risk", int(low_risk))

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Analysis Report",
                    data=csv_data,
                    file_name="fraudlens_upi_report.csv",
                    mime="text/csv"
                )

        except Exception as error:

            st.error(f"Unable to analyse file: {error}")


# ---------------------------------------
# TAB 3: MODEL INFORMATION
# ---------------------------------------

with tab3:

    st.header("About the Machine Learning Model")

    st.write(
        "FraudLens AI uses a Random Forest classifier "
        "with probability calibration."
    )

    st.subheader("Features Analysed")

    feature_descriptions = {
        "amount": "UPI transaction amount",
        "hour": "Hour of transaction",
        "transactions_last_1h": "Number of transactions in one hour",
        "new_payee": "Whether the beneficiary is new",
        "device_changed": "Whether the device has changed",
        "location_changed": "Whether the location has changed",
        "failed_upi_attempts": "Number of failed UPI attempts",
        "beneficiary_age_days": "Age of the beneficiary in days",
        "amount_vs_avg_ratio": "Transaction amount compared with usual amount"
    }

    for feature, description in feature_descriptions.items():
        st.write(f"**{feature}:** {description}")

    st.subheader("Risk Categories")

    st.write("• High Risk: 60% or above")
    st.write("• Medium Risk: 30% to below 60%")
    st.write("• Low Risk: Below 30%")

    st.caption(
        "These are illustrative display thresholds, not validated "
        "operational fraud thresholds."
    )

    st.subheader("Important Limitations")

    st.write(
        """
        - The default training dataset is synthetic.
        - The model is not connected to NPCI or any bank.
        - Model scores are not validated real-world fraud probabilities.
        - The tool does not block payments.
        - Human investigation and independent verification are required.
        """
    )


# ---------------------------------------
# FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "FraudLens AI | Academic FinTech Forensics Project | "
    "UPI & Digital Payment Fraud Detection"
)
