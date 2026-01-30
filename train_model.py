import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
credit_card_data = pd.read_csv(r"C:\Users\swaji\Desktop\project1\creditcard.csv")

# Preview dataset
print(credit_card_data.head())
print(credit_card_data.tail())

# Dataset info & missing values
credit_card_data.info()
print(credit_card_data.isnull().sum())

# Target class distribution
print(credit_card_data['Class'].value_counts())

# Separate legit and fraud transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# Statistical summary
print(legit.Amount.describe())
print(fraud.Amount.describe())

# Compare mean values by class
print(credit_card_data.groupby('Class').mean())

# Handle class imbalance using under-sampling
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Verify balanced dataset
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

# Split features and target
x = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']

print(x)
print(y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)

print(x.shape, X_train.shape, X_test.shape)

# Model definition with scaling
model = LogisticRegression()
LogisticRegression(max_iter=1000)
model = LogisticRegression(max_iter=1000)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

# Train model
model.fit(X_train, Y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Accuracy scores
print("Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Testing Accuracy:", accuracy_score(Y_test, test_pred))

# Save model and feature columns
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model, os.path.join(BASE_DIR, "credit_card_fraud_model.pkl"))
joblib.dump(list(X_train.columns), os.path.join(BASE_DIR, "feature_columns.pkl"))

print("\n✅ Saved: credit_card_fraud_model.pkl")
print("✅ Saved: feature_columns.pkl")
print("📁 Saved in:", BASE_DIR)
