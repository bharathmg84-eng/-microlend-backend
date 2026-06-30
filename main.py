"""
MicroLend FastAPI Backend — now backed by a real trained XGBoost model.
Run with: uvicorn main:app --reload --port 8000

IMPORTANT: run `python train_model.py` once before starting this, so that
model/microlend_model.pkl exists. The app will refuse to start otherwise.
"""

import os
import json
import sqlite3
from datetime import datetime

import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="MicroLend API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "microlend.db"

# ── LOAD THE REAL TRAINED MODEL ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "microlend_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"\n\nModel file not found at {MODEL_PATH}.\n"
        f"Run this first:  python train_model.py\n"
    )
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

# Order MUST match the column order used in train_model.py
FEATURE_LABELS = [
    "Business Vintage", "Daily UPI Sales", "Utility Bill History",
    "Supplier Repayment", "Loan-to-Revenue Ratio", "Business Type",
    "City Tier", "Loan Purpose",
]


# ── DB SETUP ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                name        TEXT    NOT NULL,
                biztype     TEXT    NOT NULL,
                city        TEXT    NOT NULL,
                age         INTEGER NOT NULL,
                upi         REAL    NOT NULL,
                bills       TEXT    NOT NULL,
                supplier    TEXT    NOT NULL,
                amount      REAL    NOT NULL,
                purpose     TEXT    NOT NULL,
                credit_score INTEGER NOT NULL,
                verdict     TEXT    NOT NULL,
                risk_band   TEXT    NOT NULL,
                approved_amount REAL NOT NULL,
                result_json TEXT    NOT NULL
            )
        """)
        conn.commit()

init_db()


# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
class ApplicationRequest(BaseModel):
    name:     str
    biztype:  str
    city:     str
    age:      int   = Field(..., ge=0, le=100)
    upi:      float = Field(..., ge=0)
    bills:    str
    supplier: str
    amount:   float = Field(..., gt=0)
    purpose:  str


# ── LOOKUP TABLES (same catalog as before — encoding + lender list) ─────────
BILL_SCORES = {
    "Always on time": 4,
    "Mostly on time (1–2 late/year)": 3,
    "Sometimes late": 2,
    "Frequently late": 1,
}
SUPPLIER_SCORES = {
    "Excellent — never delayed": 4,
    "Good — rarely delayed": 3,
    "Average — sometimes delayed": 2,
    "Poor — frequent delays": 1,
}
BIZ_TYPES = [
    "Kirana / General Store", "Medical / Pharmacy", "Bakery / Food Stall",
    "Tea Stall / Chai Shop", "Mobile Repair Shop", "Hardware Store",
    "Dairy / Milk Shop", "Tailoring / Textile", "Salon / Parlour",
    "Agri-input Dealer", "Vegetable / Fruit Vendor", "Other Micro-business",
]
PURPOSES = [
    "Stock / Inventory purchase", "Equipment / Machinery",
    "Shop renovation / expansion", "Working capital",
    "Emergency business expense", "Digital payment setup",
]
CITY_TIERS = {
    "Mumbai":1,"Delhi":1,"Bengaluru":1,"Pune":1,"Hyderabad":1,
    "Chennai":1,"Kolkata":1,"Ahmedabad":1,"Surat":2,"Jaipur":2,
    "Lucknow":2,"Davangere":2,"Hubli":2,"Mysuru":2,"Belagavi":2,
    "Patna":2,"Bhopal":2,"Nagpur":2,"Indore":2,"Coimbatore":2,
}
ALL_LENDERS = [
    {"name":"Bandhan Bank",      "emoji":"🏦","type":"SFB",         "minScore":70,"rate":"10.9% p.a.","max":200000},
    {"name":"Ujjivan SFB",       "emoji":"🏛","type":"SFB",         "minScore":65,"rate":"12.0% p.a.","max":150000},
    {"name":"Lendingkart",       "emoji":"📲","type":"Fintech NBFC","minScore":62,"rate":"16.0% p.a.","max":300000},
    {"name":"Arohan MFI",        "emoji":"🤝","type":"MFI",         "minScore":50,"rate":"20.0% p.a.","max":100000},
    {"name":"Annapurna Finance", "emoji":"🌿","type":"MFI",         "minScore":48,"rate":"21.0% p.a.","max":75000},
    {"name":"Mudra Kishore",     "emoji":"🇮🇳","type":"Govt Scheme","minScore":45,"rate":"10.5% p.a.","max":500000},
    {"name":"Mudra Shishu",      "emoji":"🇮🇳","type":"Govt Scheme","minScore":35,"rate":"8.5% p.a.", "max":50000},
    {"name":"PM SVANidhi",       "emoji":"🏘","type":"Govt Scheme", "minScore":30,"rate":"7.0% p.a.", "max":50000},
]


def build_shap_factors(shap_vals):
    factors = []
    for i, label in enumerate(FEATURE_LABELS):
        if i >= len(shap_vals):
            break
        val = float(shap_vals[i])
        factors.append({
            "label": label,
            "score": int(np.clip(50 + val * 200, 0, 100)),
            "weight": "positive" if val > 0.05 else "negative" if val < -0.05 else "neutral",
        })
    factors.sort(key=lambda x: abs(x["score"] - 50), reverse=True)
    return factors[:5]


# ── SCORING ENGINE — now a real model call instead of a formula ─────────────
def compute_score(req: ApplicationRequest) -> dict:
    bill_score     = BILL_SCORES.get(req.bills, 2)
    supplier_score = SUPPLIER_SCORES.get(req.supplier, 2)
    city_tier      = CITY_TIERS.get(req.city, 2)
    biz_enc        = BIZ_TYPES.index(req.biztype) if req.biztype in BIZ_TYPES else 5
    purpose_enc    = PURPOSES.index(req.purpose) if req.purpose in PURPOSES else 3
    daily_upi      = req.upi
    ltr            = req.amount / max(daily_upi * 26, 1)

    # Feature order MUST match train_model.py exactly
    features = np.array([[
        req.age, daily_upi, bill_score, supplier_score,
        ltr, biz_enc, city_tier, purpose_enc
    ]])

    prob = float(model.predict_proba(features)[0][1])
    credit_score = round(prob * 100)
    shap_vals = explainer.shap_values(features)[0]
    factors = build_shap_factors(shap_vals)

    verdict   = "APPROVED" if credit_score >= 70 else "CONDITIONAL" if credit_score >= 45 else "DECLINED"
    risk_band = "LOW"      if credit_score >= 70 else "MEDIUM"       if credit_score >= 45 else "HIGH"
    monthly   = daily_upi * 26

    approved_amount = 0.0
    if credit_score >= 80:   approved_amount = min(req.amount, monthly * 3.5)
    elif credit_score >= 70: approved_amount = min(req.amount, monthly * 2.5)
    elif credit_score >= 60: approved_amount = min(req.amount, monthly * 1.5)
    elif credit_score >= 45: approved_amount = min(req.amount, monthly * 0.8)
    approved_amount = round(approved_amount / 1000) * 1000

    lenders = [
        l for l in ALL_LENDERS
        if credit_score >= l["minScore"] and req.amount <= l["max"]
    ][:3]

    if verdict == "APPROVED":
        verdict_reason = (f"Model predicts a {prob*100:.1f}% repayment probability. "
                          f"Daily UPI of ₹{int(daily_upi):,}, {req.age} years vintage, "
                          f"and consistent payment behaviour deliver a credit score of "
                          f"{credit_score}/100 — qualifying for full approval.")
    elif verdict == "CONDITIONAL":
        verdict_reason = (f"Model predicts a {prob*100:.1f}% repayment probability — "
                          f"a moderate profile at {credit_score}/100. Loan can be approved "
                          f"at a reduced amount or slightly higher rate.")
    else:
        verdict_reason = (f"Model predicts only a {prob*100:.1f}% repayment probability, "
                          f"giving a score of {credit_score}/100, below the 45-point threshold. "
                          f"Improve financial discipline for 3–6 months before reapplying.")

    tip = None
    if verdict != "APPROVED":
        if bill_score < 3:
            tip = "Pay all utility bills on time for the next 3 months to boost your bill-history score."
        elif supplier_score < 3:
            tip = "Clear outstanding supplier dues and maintain timely payments for 60 days."
        elif ltr > 2:
            tip = f"Consider requesting ₹{round(monthly):,} (1 month revenue) instead — it reduces your loan-to-revenue ratio significantly."
        else:
            tip = "Maintain consistent UPI transactions daily and avoid any missed payments for 3 months."

    return {
        "creditScore":    credit_score,
        "riskBand":       risk_band,
        "verdict":        verdict,
        "approvedAmount": approved_amount,
        "verdictReason":  verdict_reason,
        "factors":        factors,
        "lenders":        lenders,
        "tip":            tip,
    }


# ── ROUTES (unchanged — frontend.html doesn't need to change) ───────────────
@app.post("/api/score")
def score_application(req: ApplicationRequest):
    result = compute_score(req)
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO applications
              (created_at, name, biztype, city, age, upi, bills, supplier,
               amount, purpose, credit_score, verdict, risk_band, approved_amount, result_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(),
            req.name, req.biztype, req.city, req.age, req.upi,
            req.bills, req.supplier, req.amount, req.purpose,
            result["creditScore"], result["verdict"], result["riskBand"],
            result["approvedAmount"], json.dumps(result)
        ))
        app_id = cur.lastrowid
        conn.commit()
    return {"id": app_id, **result}


@app.get("/api/applications")
def list_applications(limit: int = 20, offset: int = 0):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    return {"total": total, "items": [dict(r) for r in rows]}


@app.get("/api/applications/{app_id}")
def get_application(app_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    return dict(row)


@app.get("/api/stats")
def get_stats():
    with get_db() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
        approved= conn.execute("SELECT COUNT(*) FROM applications WHERE verdict='APPROVED'").fetchone()[0]
        avg_score= conn.execute("SELECT AVG(credit_score) FROM applications").fetchone()[0]
        total_disbursed= conn.execute("SELECT SUM(approved_amount) FROM applications WHERE verdict != 'DECLINED'").fetchone()[0]
    return {
        "total_applications": total,
        "approved":           approved,
        "approval_rate":      round(approved / total * 100, 1) if total else 0,
        "avg_credit_score":   round(avg_score, 1) if avg_score else 0,
        "total_disbursed":    total_disbursed or 0,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}
