"""Reusable cleaning, RFM, SQL and repeat-purchase modeling functions."""
from __future__ import annotations
import sqlite3
import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

REQUIRED_ALIASES = {
    "InvoiceNo": ["invoiceno", "invoice_no", "invoice", "invoice number", "orderid", "order_id", "transaction id"],
    "StockCode": ["stockcode", "stock_code", "productid", "product_id", "item code", "sku"],
    "Description": ["description", "product", "product name", "item"],
    "Quantity": ["quantity", "qty", "units", "items sold"],
    "InvoiceDate": ["invoicedate", "invoice_date", "date", "orderdate", "order_date", "transaction date"],
    "UnitPrice": ["unitprice", "unit_price", "price", "unit cost"],
    "CustomerID": ["customerid", "customer_id", "customer id", "clientid", "client_id"],
    "Country": ["country", "market", "region"],
}
FEATURES = ["recency_days", "orders_120d", "spend_120d", "avg_order_value_120d", "items_120d", "unique_products_120d", "customer_age_days"]

def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename common retail/order columns to the project's canonical schema."""
    if raw is None or raw.empty:
        raise ValueError("The uploaded file is empty.")
    df = raw.copy()
    lookup = {_norm(c): c for c in df.columns}
    rename = {}
    for canonical, aliases in REQUIRED_ALIASES.items():
        for alias in [canonical] + aliases:
            found = lookup.get(_norm(alias))
            if found is not None:
                rename[found] = canonical
                break
    df = df.rename(columns=rename)
    required = ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing) + ". Required columns include invoice/order ID, quantity, date and unit price.")
    for col, default in [("StockCode", "UNKNOWN"), ("Description", "Unknown product"), ("CustomerID", pd.NA), ("Country", "Unknown")]:
        if col not in df.columns:
            df[col] = default
    return df

def clean_transactions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Normalize raw transaction-line data; remove cancellations and invalid sale rows."""
    df = normalize_columns(raw)
    report = {"raw_rows": int(len(df))}
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", utc=True).dt.tz_convert(None)
    df["CustomerID"] = df["CustomerID"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    df.loc[df["CustomerID"].isin(["", "nan", "None", "<NA>"]), "CustomerID"] = pd.NA
    df["StockCode"] = df["StockCode"].astype("string").fillna("UNKNOWN").str.strip()
    df["Description"] = df["Description"].astype("string").fillna("Unknown product").str.strip()
    df["Country"] = df["Country"].astype("string").fillna("Unknown").str.strip()
    is_cancel = df["InvoiceNo"].str.upper().str.startswith("C", na=False) | (df["Quantity"] < 0)
    report["cancellation_rows"] = int(is_cancel.sum())
    valid_numeric = df["Quantity"].notna() & df["UnitPrice"].notna() & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    valid_date = df["InvoiceDate"].notna()
    valid_invoice = df["InvoiceNo"].notna() & (df["InvoiceNo"] != "")
    valid = (~is_cancel) & valid_numeric & valid_date & valid_invoice
    report["invalid_or_removed_rows"] = int((~valid).sum())
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError("No valid positive-quantity sales remained after cleaning. Check your date, quantity, price and invoice columns.")
    df["Revenue"] = df["Quantity"].astype(float) * df["UnitPrice"].astype(float)
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["CustomerID"] = df["CustomerID"].astype("string")
    report["clean_rows"] = int(len(df))
    report["rows_without_customer_id"] = int(df["CustomerID"].isna().sum())
    report["unique_invoices"] = int(df["InvoiceNo"].nunique())
    report["unique_customers"] = int(df["CustomerID"].nunique(dropna=True))
    return df.sort_values("InvoiceDate").reset_index(drop=True), report

def summary_metrics(df: pd.DataFrame) -> dict:
    orders = df.groupby("InvoiceNo", dropna=True).agg(order_revenue=("Revenue", "sum"))
    return {
        "revenue": float(df["Revenue"].sum()), "orders": int(df["InvoiceNo"].nunique()),
        "customers": int(df["CustomerID"].nunique(dropna=True)),
        "aov": float(orders["order_revenue"].mean()) if len(orders) else 0.0,
        "units": float(df["Quantity"].sum()), "date_min": df["InvoiceDate"].min(), "date_max": df["InvoiceDate"].max(),
    }

def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("InvoiceMonth", as_index=False).agg(revenue=("Revenue", "sum"), orders=("InvoiceNo", "nunique"), units=("Quantity", "sum"))
    return out.sort_values("InvoiceMonth")

def top_products(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    return (df.groupby(["StockCode", "Description"], as_index=False)
            .agg(revenue=("Revenue", "sum"), units=("Quantity", "sum"), orders=("InvoiceNo", "nunique"))
            .sort_values("revenue", ascending=False).head(limit))

def country_sales(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    return (df.groupby("Country", as_index=False).agg(revenue=("Revenue", "sum"), orders=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique"))
            .sort_values("revenue", ascending=False).head(limit))

def _quintile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Stable 1..5 score even with many duplicate values."""
    ranks = series.rank(method="first", pct=True)
    score = np.ceil(ranks * 5).clip(1, 5).astype(int)
    if not higher_is_better:
        score = 6 - score
    return score

def rfm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    customer_data = df.loc[df["CustomerID"].notna()].copy()
    if customer_data.empty:
        return pd.DataFrame(columns=["CustomerID", "RecencyDays", "Frequency", "Monetary", "R_score", "F_score", "M_score", "RFM_score", "Segment"])
    ref_date = customer_data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    agg = customer_data.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"), Frequency=("InvoiceNo", "nunique"), Monetary=("Revenue", "sum"),
        TotalUnits=("Quantity", "sum"), DistinctProducts=("StockCode", "nunique"),
    ).reset_index()
    agg["RecencyDays"] = (ref_date - agg["last_purchase"].dt.normalize()).dt.days
    agg["R_score"] = _quintile(agg["RecencyDays"], higher_is_better=False)
    agg["F_score"] = _quintile(agg["Frequency"], higher_is_better=True)
    agg["M_score"] = _quintile(agg["Monetary"], higher_is_better=True)
    agg["RFM_score"] = agg["R_score"].astype(str) + agg["F_score"].astype(str) + agg["M_score"].astype(str)
    def label(row):
        r, f = row["R_score"], row["F_score"]
        if r >= 4 and f >= 4: return "Champions"
        if r >= 3 and f >= 3: return "Loyal customers"
        if r >= 4 and f <= 2: return "Recent customers"
        if r <= 2 and f >= 3: return "At risk"
        if r <= 2 and f <= 2: return "Hibernating"
        return "Potential loyalists"
    agg["Segment"] = agg.apply(label, axis=1)
    agg = agg.rename(columns={"last_purchase": "LastPurchase"})
    return agg.sort_values(["Monetary", "Frequency"], ascending=False).reset_index(drop=True)

def build_customer_snapshots(df: pd.DataFrame, lookback_days: int = 120, horizon_days: int = 30, step_days: int = 30) -> pd.DataFrame:
    """Build historical point-in-time customer snapshots with a future-window repeat-purchase label."""
    tx = df[df["CustomerID"].notna()].copy()
    if tx.empty:
        return pd.DataFrame()
    tx["CustomerID"] = tx["CustomerID"].astype(str)
    tx = tx.sort_values("InvoiceDate")
    first_date, last_date = tx["InvoiceDate"].min().normalize(), tx["InvoiceDate"].max().normalize()
    cutoffs = pd.date_range(first_date + pd.Timedelta(days=lookback_days), last_date - pd.Timedelta(days=horizon_days), freq=f"{step_days}D")
    parts = []
    for cutoff in cutoffs:
        historical = tx[tx["InvoiceDate"] < cutoff]
        future = tx[(tx["InvoiceDate"] >= cutoff) & (tx["InvoiceDate"] < cutoff + pd.Timedelta(days=horizon_days))]
        if historical.empty:
            continue
        lifetime = historical.groupby("CustomerID").agg(first_purchase=("InvoiceDate", "min"), last_purchase=("InvoiceDate", "max"), lifetime_orders=("InvoiceNo", "nunique"))
        recent = historical[historical["InvoiceDate"] >= cutoff - pd.Timedelta(days=lookback_days)]
        if recent.empty:
            recent_agg = pd.DataFrame(index=lifetime.index, columns=["orders_120d", "spend_120d", "items_120d", "unique_products_120d", "avg_order_value_120d"])
        else:
            rec_inv = recent.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(order_revenue=("Revenue", "sum"))
            recent_agg = recent.groupby("CustomerID").agg(orders_120d=("InvoiceNo", "nunique"), spend_120d=("Revenue", "sum"), items_120d=("Quantity", "sum"), unique_products_120d=("StockCode", "nunique"))
            recent_agg["avg_order_value_120d"] = rec_inv.groupby("CustomerID")["order_revenue"].mean()
        snap = lifetime.join(recent_agg, how="left")
        snap["recency_days"] = (cutoff - snap["last_purchase"]).dt.total_seconds() / 86400
        snap["customer_age_days"] = (cutoff - snap["first_purchase"]).dt.total_seconds() / 86400
        for col in ["orders_120d", "spend_120d", "items_120d", "unique_products_120d", "avg_order_value_120d"]:
            snap[col] = pd.to_numeric(snap[col], errors="coerce").fillna(0.0)
        future_customers = set(future["CustomerID"].astype(str).unique())
        snap["target_repeat_30d"] = snap.index.astype(str).isin(future_customers).astype(int)
        snap["cutoff"] = cutoff
        snap.index.name = "CustomerID"
        snap = snap.reset_index()
        parts.append(snap[["CustomerID", "cutoff"] + FEATURES + ["target_repeat_30d"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def train_repeat_purchase_model(df: pd.DataFrame) -> dict:
    """Train and evaluate a logistic model using a chronological cutoff split."""
    snapshots = build_customer_snapshots(df)
    if snapshots.empty or snapshots["target_repeat_30d"].nunique() < 2 or snapshots["cutoff"].nunique() < 5:
        return {"available": False, "reason": "There is not enough dated customer history to create at least five historical snapshots with both target classes. Use a longer date range or more transactions."}
    cutoffs = sorted(snapshots["cutoff"].unique())
    split_index = max(1, int(len(cutoffs) * .8))
    train_cutoffs, test_cutoffs = set(cutoffs[:split_index]), set(cutoffs[split_index:])
    train, test = snapshots[snapshots["cutoff"].isin(train_cutoffs)], snapshots[snapshots["cutoff"].isin(test_cutoffs)]
    if train["target_repeat_30d"].nunique() < 2 or test["target_repeat_30d"].nunique() < 2:
        snapshots = snapshots.sort_values("cutoff")
        boundary = int(len(snapshots) * .8)
        train, test = snapshots.iloc[:boundary], snapshots.iloc[boundary:]
    if train.empty or test.empty or train["target_repeat_30d"].nunique() < 2:
        return {"available": False, "reason": "The time-based holdout does not contain enough examples of both classes to evaluate a model reliably."}
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    model.fit(train[FEATURES], train["target_repeat_30d"])
    probs, preds = model.predict_proba(test[FEATURES])[:, 1], None
    preds = (probs >= .5).astype(int)
    y = test["target_repeat_30d"].astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)), "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)), "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)) if y.nunique() == 2 else None,
        "test_rows": int(len(test)), "train_rows": int(len(train)), "test_positive_rate": float(y.mean()),
        "split": "Chronological holdout by snapshot cutoff",
    }
    coefs = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.named_steps["classifier"].coef_[0]})
    coefs["AbsCoefficient"] = coefs["Coefficient"].abs()
    coefs = coefs.sort_values("AbsCoefficient", ascending=False).drop(columns="AbsCoefficient")
    cm = confusion_matrix(y, preds, labels=[0, 1])
    cutoff = df["InvoiceDate"].max() + pd.Timedelta(seconds=1)
    tx = df[df["CustomerID"].notna()].copy()
    tx["CustomerID"] = tx["CustomerID"].astype(str)
    lifetime = tx.groupby("CustomerID").agg(first_purchase=("InvoiceDate", "min"), last_purchase=("InvoiceDate", "max"))
    recent = tx[tx["InvoiceDate"] >= cutoff - pd.Timedelta(days=120)]
    if recent.empty:
        recent_agg = pd.DataFrame(index=lifetime.index, columns=["orders_120d", "spend_120d", "items_120d", "unique_products_120d", "avg_order_value_120d"])
    else:
        rec_inv = recent.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(order_revenue=("Revenue", "sum"))
        recent_agg = recent.groupby("CustomerID").agg(orders_120d=("InvoiceNo", "nunique"), spend_120d=("Revenue", "sum"), items_120d=("Quantity", "sum"), unique_products_120d=("StockCode", "nunique"))
        recent_agg["avg_order_value_120d"] = rec_inv.groupby("CustomerID")["order_revenue"].mean()
    current = lifetime.join(recent_agg, how="left")
    current["recency_days"] = (cutoff - current["last_purchase"]).dt.total_seconds() / 86400
    current["customer_age_days"] = (cutoff - current["first_purchase"]).dt.total_seconds() / 86400
    for col in FEATURES:
        if col not in current.columns:
            current[col] = 0.0
        current[col] = pd.to_numeric(current[col], errors="coerce").fillna(0.0)
    current["repeat_probability"] = model.predict_proba(current[FEATURES])[:, 1]
    current = current.reset_index().rename(columns={"index": "CustomerID"})
    current["CustomerID"] = current["CustomerID"].astype(str)
    current["RiskBand"] = pd.cut(current["repeat_probability"], bins=[-0.001, .33, .66, 1.001], labels=["Lower propensity", "Medium propensity", "Higher propensity"])
    return {"available": True, "model": model, "snapshots": snapshots, "metrics": metrics, "feature_coefficients": coefs, "confusion_matrix": cm, "predictions": current.sort_values("repeat_probability", ascending=False), "test_results": pd.DataFrame({"Actual": y.to_numpy(), "Predicted": preds, "Probability": probs}), "feature_names": FEATURES}

def run_sql_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Run a read-only SELECT / WITH query against an in-memory SQLite table named transactions."""
    cleaned = query.strip().lower()
    if not (cleaned.startswith("select") or cleaned.startswith("with")):
        raise ValueError("Only read-only SELECT / WITH statements are allowed in the SQL explorer.")
    with sqlite3.connect(":memory:") as conn:
        df.to_sql("transactions", conn, index=False, if_exists="replace")
        return pd.read_sql_query(query, conn)
