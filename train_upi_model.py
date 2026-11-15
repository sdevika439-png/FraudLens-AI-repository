
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

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


def generate_demo_data(n=12000, random_state=42):
    """
    Generate artificial UPI-like transactions
    for educational model training.

    This is NOT real bank or NPCI data.
    """

    rng = np.random.default_rng(random_state)

    amount = np.clip(
        rng.lognormal(mean=7.0, sigma=1.2, size=n),
        10, 200000
    )

    hour = rng.integers(0, 24, size=n)

    transactions_last_1h = rng.poisson(1.5, size=n)

    new_payee = rng.binomial(1, 0.20, size=n)
    device_changed = rng.binomial(1, 0.10, size=n)
    location_changed = rng.binomial(1, 0.08, size=n)

    failed_upi_attempts = rng.poisson(0.3, size=n)

    beneficiary_age_days = rng.integers(0, 1000, size=n)

    amount_vs_avg_ratio = np.clip(
        rng.lognormal(mean=0, sigma=0.8, size=n),
        0.1, 15
    )

    # Artificial teaching labels.
    # These rules create a synthetic fraud pattern.
    risk_signal = (
        0.9 * (amount > 15000)
        + 0.8 * (hour <= 5)
        + 0.9 * (transactions_last_1h >= 5)
        + 1.0 * new_payee
        + 0.9 * device_changed
        + 0.7 * location_changed
        + 1.0 * (failed_upi_attempts >= 2)
        + 0.8 * (beneficiary_age_days <= 2)
        + 0.8 * (amount_vs_avg_ratio >= 4)
    )

    fraud_probability = 1 / (
        1 + np.exp(-(risk_signal - 4.2))
    )

    fraud = rng.binomial(1, fraud_probability)

    data = pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "transactions_last_1h": transactions_last_1h,
        "new_payee": new_payee,
        "device_changed": device_changed,
        "location_changed": location_changed,
        "failed_upi_attempts": failed_upi_attempts,
        "beneficiary_age_days": beneficiary_age_days,
        "amount_vs_avg_ratio": amount_vs_avg_ratio,
        "fraud": fraud
    })

    return data


def main():

    print("\n FRAUDLENS AI - UPI MODEL TRAINING")
    print("-----------------------------------")

    # Load a labelled dataset if supplied.
    # Otherwise, generate artificial teaching data.
    try:
        data = pd.read_csv("upi_train.csv")

        required = FEATURES + ["fraud"]

        missing = [
            column for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        data = data[required].copy()
        print("Using supplied labelled dataset: upi_train.csv")

    except FileNotFoundError:
        print("upi_train.csv not found.")
        print("Generating artificial UPI-like demo data.")
        data = generate_demo_data()

    X = data[FEATURES].copy()
    y = pd.to_numeric(data["fraud"], errors="coerce")

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("The fraud column must contain only 0 and 1.")

    if y.nunique() != 2:
        raise ValueError("Training data must contain both fraud classes.")

    if len(data) < 20:
        raise ValueError("Please provide at least 20 labelled transactions.")

    X = X.apply(pd.to_numeric, errors="coerce")

    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Machine learning model.
    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    # Calibrate model outputs.
    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3
    )

    print("\nTraining model...")

    model.fit(X_train, y_train)

    # Evaluate the model.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions))

    print("\nROC-AUC:",
          round(roc_auc_score(y_test, probabilities), 4))

    print("Average Precision:",
          round(average_precision_score(y_test, probabilities), 4))

    # Save the trained model.
    joblib.dump(model, MODEL_FILE)

    print("\nModel successfully saved as:", MODEL_FILE)
    print("Training completed!")


if __name__ == "__main__":
    main()
