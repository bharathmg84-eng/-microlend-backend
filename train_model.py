"""
train_model.py - STRONG VERSION
Generates realistic synthetic data with strong signal and trains XGBoost.
Expected AUC: 0.82-0.87
Run: python train_model.py
Model saves to: model/microlend_model.pkl
"""

import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

np.random.seed(42)
N = 8000

# ── Generate realistic MSME applicant data ────────────────────────────────────
business_age      = np.random.exponential(scale=4, size=N).clip(0, 30)
daily_upi_sales   = np.random.lognormal(mean=8.2, sigma=0.6, size=N).clip(300, 20000)
bill_score        = np.random.choice([1, 2, 3, 4], size=N, p=[0.10, 0.20, 0.30, 0.40])
supplier_score    = np.random.choice([1, 2, 3, 4], size=N, p=[0.10, 0.22, 0.33, 0.35])
biz_enc           = np.random.randint(0, 12, size=N)
city_tier         = np.random.choice([1, 2], size=N, p=[0.45, 0.55])
purpose_enc       = np.random.randint(0, 6, size=N)
requested_loan    = np.random.lognormal(mean=10.8, sigma=0.5, size=N).clip(10000, 500000)
monthly_revenue   = daily_upi_sales * 26
ltr               = (requested_loan / np.maximum(monthly_revenue, 1)).clip(0, 10)

# ── Strong signal — scale=2.2 gives AUC ~0.85 ────────────────────────────────
SIGNAL_SCALE = 2.2
z = SIGNAL_SCALE * (
    1.00 * (bill_score - 2.5) / 1.5
    + 0.90 * (supplier_score - 2.5) / 1.5
    + 0.80 * np.tanh((business_age - 3) / 4)
    + 0.70 * np.tanh((daily_upi_sales - 3000) / 3000)
    - 1.10 * np.tanh(ltr / 2)
    + 0.20 * (city_tier == 1).astype(float)
) + np.random.normal(0, 0.45, size=N)

prob_repay = 1 / (1 + np.exp(-z))
repaid = np.random.binomial(1, prob_repay)

# Feature order MUST match main.py exactly
X = np.column_stack([
    business_age, daily_upi_sales, bill_score, supplier_score,
    ltr, biz_enc, city_tier, purpose_enc
])
y = repaid

print(f"Dataset: {N} rows | Base repayment rate: {y.mean():.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred_proba)
acc = accuracy_score(y_test, (pred_proba >= 0.5).astype(int))

print(f"Validation AUC-ROC : {auc:.3f}")
print(f"Validation Accuracy: {acc:.3f}")

out_dir = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "microlend_model.pkl")
joblib.dump(model, out_path)
print(f"Model saved to {out_path}")
