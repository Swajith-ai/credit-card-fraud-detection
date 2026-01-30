# Credit Card Fraud Detection using Machine Learning

This project implements a Credit Card Fraud Detection system using
Logistic Regression.

The dataset used is highly imbalanced and contains a very small number
of fraudulent transactions compared to legitimate ones.

To handle this, under-sampling was applied to create a balanced dataset
before training the model.

## Features
- Binary classification (Fraud / Legit)
- Logistic Regression model
- Feature scaling using StandardScaler
- Model persistence using joblib
- Separate prediction script

## Files in this Repository
- train_model.py : Trains the machine learning model
- fraud_predictor.py : Loads the trained model and predicts fraud
- credit_card_fraud_model.pkl : Saved trained model
- feature_columns.pkl : Feature list used during training

## Dataset
The dataset is not included in this repository due to GitHub file size limits.

Dataset source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Run predictions:
   python fraud_predictor.py

## Author
Swajith S S
