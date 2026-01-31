import streamlit as st
import pandas as pd
from predict_service import feature_columns, predict_from_values, predict_from_csv_df

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction features and get a fraud prediction with probability.")

tab1, tab2, tab3 = st.tabs(["🧾 Manual Input", "📋 Paste Values", "📁 CSV Upload"])

# -------------------------
# TAB 1: Manual Input
# -------------------------
with tab1:
    st.subheader("Manual input (one by one)")

    with st.form("manual_form"):
        cols = st.columns(3)
        values = []

        for i, col in enumerate(feature_columns):
            with cols[i % 3]:
                v = st.number_input(col, value=0.0, format="%.6f")
                values.append(float(v))

        submitted = st.form_submit_button("Predict")

        if submitted:
            pred, prob = predict_from_values(values)
            if pred == 1:
                st.error(f"🚨 FRAUD TRANSACTION (probability = {prob:.4f})")
            else:
                st.success(f"✅ LEGIT TRANSACTION (probability = {prob:.4f})")

# -------------------------
# TAB 2: Paste Values
# -------------------------
with tab2:
    st.subheader("Paste comma-separated values")
    st.caption("Order must match the feature columns below:")

    st.code(", ".join(feature_columns))

    raw = st.text_area(
        "Paste values here",
        height=150,
        placeholder="value1, value2, value3, ..."
    )

    if st.button("Predict from pasted values"):
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]

        if len(parts) != len(feature_columns):
            st.warning(f"Expected {len(feature_columns)} values but got {len(parts)}")
        else:
            try:
                values = [float(p) for p in parts]
                pred, prob = predict_from_values(values)

                if pred == 1:
                    st.error(f"🚨 FRAUD TRANSACTION (probability = {prob:.4f})")
                else:
                    st.success(f"✅ LEGIT TRANSACTION (probability = {prob:.4f})")

            except ValueError:
                st.error("❌ Invalid number found. Please check your pasted values.")

# -------------------------
# TAB 3: CSV Upload
# -------------------------
with tab3:
    st.subheader("Upload CSV for batch prediction")
    st.caption("CSV must contain all required feature columns (exact names).")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    st.write("Required columns:")
    st.code(", ".join(feature_columns))

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, skipinitialspace=True)
            df.columns = df.columns.str.strip()

            st.write("Preview of uploaded file:")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Run batch prediction"):
                result_df = predict_from_csv_df(df)

                st.success("✅ Prediction completed!")
                st.write("Preview of results:")
                st.dataframe(result_df.head(10), use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download results CSV",
                    data=csv_bytes,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
