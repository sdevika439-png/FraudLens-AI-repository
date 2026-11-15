# FraudLens AI – UPI & Digital Payment Fraud Detection

An academic machine learning project designed to explore suspicious transaction patterns in India's digital payment ecosystem.

## Project Overview

FraudLens AI is a Python-based fraud detection prototype that analyses UPI-like transaction characteristics and generates risk scores to support preliminary fraud investigation.

The project explores how machine learning can assist in identifying unusual transaction behaviour, potential account misuse and suspicious payment patterns.

## Key Features

- Single UPI transaction analysis
- Bulk transaction analysis using CSV files
- Machine learning-based risk scoring
- Low, Medium and High risk categories
- Identification of suspicious transaction indicators
- Downloadable transaction analysis reports
- Interactive Streamlit dashboard

## Fraud Indicators

The prototype considers:

- Unusually high transaction amounts
- Transactions at unusual hours
- Multiple transactions within a short period
- Newly added beneficiaries
- Changes in device information
- Changes in location information
- Repeated failed UPI attempts
- Recently created beneficiaries
- Transaction amounts significantly above usual spending

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming |
| Streamlit | Dashboard |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Random Forest | Classification |
| Joblib | Model storage |

## Installation

Clone the repository:

```bash
git clone https://github.com/sdevika439-png/FraudLens-AI.git
cd FraudLens-AI
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Train the model:

```bash
python3 train_upi_model.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Project Structure

```text
FraudLens-AI/
│
├── app.py
├── train_upi_model.py
├── requirements.txt
├── sample_upi_transactions.csv
├── README.md
└── .gitignore
```

## Important Disclaimer

This project is an academic and portfolio demonstration.

The model uses synthetic UPI-like data and does not use actual NPCI or bank transaction records. Its risk scores are experimental outputs, not validated real-world fraud probabilities.

FraudLens AI is not connected to live UPI infrastructure, NPCI systems or banking APIs. It does not block payments or establish that a customer has committed fraud.

All sample data is fictional. No UPI PINs, OTPs, passwords or banking credentials should be collected or uploaded.

## Future Improvements

- Mule account detection
- Transaction network analysis
- Explainable AI using SHAP
- Improved anomaly detection
- Evaluation using suitable labelled datasets
- Investigation-focused fraud pattern visualisation

## Author

**Devika Santhosh**

FinTech Forensics | Financial Crime | AI & Digital Payment Security

Academic portfolio project.
