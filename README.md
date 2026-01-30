# Credit Card Fraud Detection using Machine Learning

This project implements a Credit Card Fraud Detection System using a Logistic Regression
machine learning model.

The dataset used in this project is highly imbalanced, meaning fraudulent transactions
are extremely rare compared to legitimate ones.

To handle this problem correctly, under-sampling is applied to create a balanced dataset
before training the model.

The trained model is saved and reused for prediction using a separate prediction script.

## Problem Statement

In real-world credit card transactions, legitimate transactions vastly outnumber
fraudulent ones.

Training a machine learning model directly on such imbalanced data can lead to biased
predictions.

In some cases, a model may achieve nearly 99% accuracy by predicting all transactions
as legitimate, which is misleading and ineffective.

This project focuses on detecting fraudulent transactions accurately rather than
achieving inflated accuracy scores.

## Solution Approach

The solution follows a structured machine learning workflow.

The dataset is loaded and explored to understand its structure and distribution.
Legitimate and fraudulent transactions are separated, and an under-sampling technique
is applied to balance the dataset.

A Logistic Regression model is then trained using feature scaling to ensure numerical
stability and proper convergence.

The model is evaluated on both training and testing datasets, saved for reuse, and
applied to predict fraud on new transactions.

## Key Features

- Binary classification where:
  - Class `0` represents legitimate transactions
  - Class `1` represents fraudulent transactions
- Handles class imbalance using under-sampling
- Feature scaling using StandardScaler within an sklearn Pipeline
- Prevents convergence warnings during training
- Model persistence using joblib
- Interactive prediction system
- Supports manual input, pasted values, and CSV-based predictions

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Dataset

The project uses the Credit Card Fraud Detection dataset published by ULB and hosted
on Kaggle.

- File name: `creditcard.csv`
- Approximate size: 150 MB
- Not included in this repository due to GitHub’s 100 MB file size limit

Dataset source:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Dataset Characteristics

The dataset contains:
- 284,315 legitimate transactions
- 492 fraudulent transactions

This severe class imbalance makes direct training ineffective without applying
balancing techniques.

## Handling Class Imbalance

To mitigate the impact of class imbalance, an under-sampling strategy is used.

A random sample of 492 legitimate transactions is selected and combined with all
492 fraudulent transactions.

This results in a balanced dataset containing an equal number of legitimate and
fraudulent transactions, enabling the model to learn meaningful patterns for both
classes.

## Project Workflow

1. Import required dependencies
2. Load the CSV dataset into a Pandas DataFrame
3. Explore the dataset and check for missing values
4. Analyze transaction distribution
5. Separate legitimate and fraudulent transactions
6. Perform statistical analysis
7. Apply under-sampling
8. Create a balanced dataset
9. Split data into training and testing sets
10. Scale features using StandardScaler
11. Train the Logistic Regression model
12. Evaluate performance using accuracy metrics
13. Save the trained model and feature columns
14. Perform predictions on new data

## Project Files

- `train_model.py`  
  Handles data preprocessing, model training, evaluation, and saving the trained model

- `fraud_predictor.py`  
  Loads the saved model and performs fraud prediction on new inputs

- `credit_card_fraud_model.pkl`  
  Saved trained model

- `feature_columns.pkl`  
  Feature column order used during training

- `README.md`  
  Project documentation

-
If i make any changes , i should type the below things in terminal to push the changes to git repo
git add .
git commit -m "Describe what you changed"
git push

## Author

Swajith S S
