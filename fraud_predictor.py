import pandas as pd
import joblib

model = joblib.load("credit_card_fraud_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

print("✅ Model loaded successfully")

def predict_one(input_data):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data], columns=feature_columns)
    else:
        if len(input_data) != len(feature_columns):
            raise ValueError(f"Expected {len(feature_columns)} values")
        df = pd.DataFrame([input_data], columns=feature_columns)

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return pred, prob

def show_result(pred, prob):
    if pred == 1:
        print(f"\n🚨 FRAUD TRANSACTION (probability = {prob:.4f})")
    else:
        print(f"\n✅ LEGIT TRANSACTION (probability = {prob:.4f})")

def mode_manual():
    print("\nEnter values one by one")
    values = []
    for col in feature_columns:
        while True:
            try:
                v = float(input(f"{col}: "))
                values.append(v)
                break
            except ValueError:
                print("Please enter a valid number.")
    pred, prob = predict_one(values)
    show_result(pred, prob)

def mode_paste():
    print("\nPaste comma-separated values")
    print(", ".join(feature_columns))

    raw = input("\nPaste values: ").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]

    if len(parts) != len(feature_columns):
        print(f"\n❌ Expected {len(feature_columns)} values but got {len(parts)}")
        return

    try:
        values = [float(p) for p in parts]
    except ValueError:
        print("\n❌ Invalid number found")
        return

    pred, prob = predict_one(values)
    show_result(pred, prob)

def mode_csv():
    print("\nPredicting from CSV file")
    print(", ".join(feature_columns))

    path = input("\nEnter CSV file path: ").strip().strip('"').strip("'")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("\n❌ Could not read file:", e)
        return

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print("\n❌ Missing columns:", missing)
        return

    X = df[feature_columns]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["Predicted_Class"] = preds
    df["Fraud_Probability"] = probs

    print("\nFirst 10 results:\n")
    print(df[["Predicted_Class", "Fraud_Probability"]].head(10))

    save = input("\nSave results to CSV? (y/n): ").lower()
    if save == "y":
        out = input("Output file name: ")
        df.to_csv(out, index=False)
        print("✅ Results saved")

while True:
    print("\n1) Manual input")
    print("2) Paste comma-separated values")
    print("3) Predict from CSV")
    print("0) Exit")

    choice = input("Choice: ").strip()

    if choice == "1":
        mode_manual()
    elif choice == "2":
        mode_paste()
    elif choice == "3":
        mode_csv()
    elif choice == "0":
        print("Exiting...")
        break
    else:
        print("❌ Invalid choice")
