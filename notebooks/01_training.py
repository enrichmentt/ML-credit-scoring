#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
import os
import json


# In[2]:


DATA_PATH = '../data/raw/UCI_Credit_Card.csv'
df = pd.read_csv(DATA_PATH)

df = df.rename(columns={'default.payment.next.month': 'target'})

df = df.drop('ID', axis=1)

print(f"Размер датасета: {df.shape}")
df.head()


# In[3]:


X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")


# In[4]:


model_v1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_v1.fit(X_train, y_train)

preds_v1 = model_v1.predict(X_test)


# In[5]:


print("V1")
print(classification_report(y_test, preds_v1))

f1_v1 = f1_score(y_test, preds_v1)
precision_v1 = precision_score(y_test, preds_v1)
recall_v1 = recall_score(y_test, preds_v1)

print(f"F1 Score: {f1_v1:.4f}")
print(f"Precision: {precision_v1:.4f}")
print(f"Recall: {recall_v1:.4f}")


# In[6]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_v2 = LogisticRegression(max_iter=1000, random_state=42)
model_v2.fit(X_train_scaled, y_train)

preds_v2 = model_v2.predict(X_test_scaled)

print("--- Модель v2 (Тестовая) ---")
print(classification_report(y_test, preds_v2))

f1_v2 = f1_score(y_test, preds_v2)
precision_v2 = precision_score(y_test, preds_v2)
recall_v2 = recall_score(y_test, preds_v2)


# In[7]:


os.makedirs('../models', exist_ok=True)


joblib.dump(model_v1, '../models/model_v1.joblib')

joblib.dump(model_v2, '../models/model_v2.joblib')
joblib.dump(scaler, '../models/scaler_v2.joblib')


feature_names = X.columns.tolist()
with open('../models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print("Save on ../models/")


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)


scaler_v2 = joblib.load('../models/scaler_v2.joblib')
model_v1 = joblib.load('../models/model_v1.joblib')
model_v2 = joblib.load('../models/model_v2.joblib')

preds_v1 = model_v1.predict(X_test)
preds_v2 = model_v2.predict(scaler_v2.transform(X_test))

sns.set_theme(style="whitegrid")

# ROC-AUC
proba_v1 = model_v1.predict_proba(X_test)[:, 1]
roc_auc_v1 = roc_auc_score(y_test, proba_v1)

proba_v2 = model_v2.predict_proba(scaler_v2.transform(X_test))[:, 1]
roc_auc_v2 = roc_auc_score(y_test, proba_v2)

print(f"ROC-AUC модели v1 (Случайный лес): {roc_auc_v1:.4f}")
print(f"ROC-AUC модели v2 (Логистическая регрессия): {roc_auc_v2:.4f}")
print("-" * 50)

# ошибки
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_v1 = confusion_matrix(y_test, preds_v1)
sns.heatmap(cm_v1, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Матрица ошибок: Модель v1')
axes[0].set_xlabel('Предсказано')
axes[0].set_ylabel('Реальность')
axes[0].set_yticklabels(['Не дефолт (0)', 'Дефолт (1)'], rotation=0)
axes[0].set_xticklabels(['Не дефолт (0)', 'Дефолт (1)'])

cm_v2 = confusion_matrix(y_test, preds_v2)
sns.heatmap(cm_v2, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title('Матрица ошибок: Модель v2')
axes[1].set_xlabel('Предсказано')
axes[1].set_ylabel('Реальность')
axes[1].set_yticklabels(['Не дефолт (0)', 'Дефолт (1)'], rotation=0)
axes[1].set_xticklabels(['Не дефолт (0)', 'Дефолт (1)'])

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, proba_v1, name=f"Модель v1 (AUC={roc_auc_v1:.3f})", ax=ax)
RocCurveDisplay.from_predictions(y_test, proba_v2, name=f"Модель v2 (AUC={roc_auc_v2:.3f})", ax=ax)

ax.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")
ax.set_title("ROC-кривая: Сравнение моделей")
ax.legend(loc="lower right")
plt.show()
