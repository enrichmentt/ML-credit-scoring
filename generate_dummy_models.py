import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import os

os.makedirs('models', exist_ok=True)

X = pd.DataFrame([
    {
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    },
    {
        "LIMIT_BAL": 10000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 30,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        "BILL_AMT1": 0, "BILL_AMT2": 0, "BILL_AMT3": 0, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 0,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    },
    {
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
        "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689,
        "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
    }
])
y = [1, 0, 1]

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)

lr = LogisticRegression(random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr.fit(X_scaled, y)

joblib.dump(rf, 'models/model_v1.joblib')
joblib.dump(lr, 'models/model_v2.joblib')
joblib.dump(scaler, 'models/scaler_v2.joblib')

with open('models/feature_names.json', 'w') as f:
    json.dump(list(X.columns), f)

print("Dummy models generated successfully.")
