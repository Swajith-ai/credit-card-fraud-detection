Credit Card Fraud Detection using Machine Learning

This project implements a Credit Card Fraud Detection System using a Logistic Regression machine learning model.

The dataset used in this project is highly imbalanced, meaning fraudulent transactions are extremely rare compared to legitimate ones. To handle this problem correctly, under-sampling is applied to create a balanced dataset before training the model.

The trained model is saved and reused for prediction using both a command-line interface and a user-friendly web-based interface built with Streamlit.

Problem Statement

In real-world credit card transactions, legitimate transactions vastly outnumber fraudulent ones. Training a machine learning model directly on such imbalanced data can lead to biased predictions.

In some cases, a model may achieve nearly 99% accuracy by predicting all transactions as legitimate, which is misleading and ineffective.

This project focuses on detecting fraudulent transactions accurately, rather than achieving inflated accuracy scores.

Solution Approach

The solution follows a structured machine learning workflow.

The dataset is loaded and explored to understand its structure and distribution. Legitimate and fraudulent transactions are separated, and an under-sampling technique is applied to balance the dataset.

A Logistic Regression model is then trained using feature scaling to ensure numerical stability and proper convergence.

The model is evaluated on both training and testing datasets, saved for reuse, and applied to predict fraud on new transactions using an interactive user interface.

Key Features

Binary classification

Class 0 → Legitimate transaction

Class 1 → Fraudulent transaction

Handles class imbalance using under-sampling

Feature scaling using StandardScaler within an sklearn Pipeline

Prevents convergence warnings during training

Model persistence using joblib

Interactive web-based UI using Streamlit

Supports:

Manual input

Paste comma-separated values

CSV-based batch predictions with downloadable results

Tech Stack

Python

NumPy

Pandas

Scikit-learn

Joblib

Streamlit

Dataset

The project uses the Credit Card Fraud Detection dataset published by ULB and hosted on Kaggle.

File name: creditcard.csv

Approximate size: 150 MB

Note: Not included in this repository due to GitHub’s 100 MB file size limit

Dataset source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset Characteristics

284,315 legitimate transactions

492 fraudulent transactions

This severe class imbalance makes direct training ineffective without applying balancing techniques.

Handling Class Imbalance

To mitigate the impact of class imbalance, an under-sampling strategy is used.

A random sample of 492 legitimate transactions is selected and combined with all 492 fraudulent transactions, creating a balanced dataset.

This enables the model to learn meaningful patterns for both classes.

Project Workflow

Import required dependencies

Load the CSV dataset into a Pandas DataFrame

Explore the dataset and check for missing values

Analyze transaction distribution

Separate legitimate and fraudulent transactions

Perform statistical analysis

Apply under-sampling

Create a balanced dataset

Split data into training and testing sets

Scale features using StandardScaler

Train the Logistic Regression model

Evaluate performance using accuracy metrics

Save the trained model and feature columns

Perform predictions using a user interface

Project Files

train_model.py — Model training and evaluation

predict_service.py — Reusable prediction logic

app.py — Streamlit-based user interface

credit_card_fraud_model.pkl — Saved trained model

feature_columns.pkl — Feature column order

requirements.txt — Python dependencies

README.md — Project documentation

Run the User Interface (Streamlit)
Install dependencies
pip install -r requirements.txt

Run the application
python -m streamlit run app.py

Open in browser (if not automatic)
http://localhost:8501

Updating the Repository
git add .
git commit -m "Describe what you changed"
git push

Author

Swajith S S