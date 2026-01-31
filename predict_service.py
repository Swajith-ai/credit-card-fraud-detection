import os
import joblib
import pandas as pd

# Always load model files from the same folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "credit_card_fraud_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

def predict_from_values(values):
    """
    values: list of floats in the exact order of feature_columns
    returns: (pred_class_int, fraud_probability_float)
    """
    if len(values) != len(feature_columns):
        raise ValueError(f"Expected {len(feature_columns)} values but got {len(values)}")

    df = pd.DataFrame([values], columns=feature_columns)
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return pred, prob


def predict_from_csv_df(df):
    """
    df: pandas DataFrame that contains feature_columns
    returns: df_copy with Predicted_Class and Fraud_Probability
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_columns]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["Predicted_Class"] = preds
    out["Fraud_Probability"] = probs
    return out
