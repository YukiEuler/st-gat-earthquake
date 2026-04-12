"""
Revision helper: domain-centric hazard utility metrics.

Reads:
    outputs/predictions.csv

Writes:
    outputs/revision_analysis/hazard_utility_by_horizon.csv
    outputs/revision_analysis/hazard_utility_summary.json

Metrics:
- precision/recall/F1/specificity for significant-event detection (Mw >= threshold)
- lead-time proxy from earliest predicted alert vs earliest observed event per node/sample
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def run(threshold: float = 1.0) -> None:
    root = Path(__file__).resolve().parents[1]
    in_csv = root / "outputs" / "predictions.csv"
    out_dir = root / "outputs" / "revision_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}")

    df = pd.read_csv(in_csv)
    required = {"sample_idx", "horizon", "node_id", "max_mw_target", "max_mw_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")

    # Per-horizon classification utility
    rows = []
    for h, g in df.groupby("horizon"):
        y_true = (g["max_mw_target"].to_numpy() >= threshold).astype(int)
        y_pred = (g["max_mw_pred"].to_numpy() >= threshold).astype(int)
        m = confusion_metrics(y_true, y_pred)
        m.update({"horizon": int(h), "n": int(len(g)), "threshold": float(threshold)})
        rows.append(m)

    by_h = pd.DataFrame(rows).sort_values("horizon")
    out_h_csv = out_dir / "hazard_utility_by_horizon.csv"
    by_h.to_csv(out_h_csv, index=False)

    # Overall metrics
    y_true_all = (df["max_mw_target"].to_numpy() >= threshold).astype(int)
    y_pred_all = (df["max_mw_pred"].to_numpy() >= threshold).astype(int)
    overall = confusion_metrics(y_true_all, y_pred_all)

    # Lead-time proxy (if multi-horizon predictions are available)
    lead_records = []
    for (_, _), g in df.sort_values("horizon").groupby(["sample_idx", "node_id"]):
        true_mask = g["max_mw_target"].to_numpy() >= threshold
        pred_mask = g["max_mw_pred"].to_numpy() >= threshold
        h = g["horizon"].to_numpy()

        if np.any(true_mask):
            h_event = int(h[np.argmax(true_mask)])
            if np.any(pred_mask):
                h_alert = int(h[np.argmax(pred_mask)])
                lead_records.append(h_event - h_alert)  # >0 early, 0 on-time, <0 late

    if lead_records:
        lead = np.array(lead_records)
        lead_summary = {
            "n_event_cases": int(len(lead)),
            "mean_lead_horizon_steps": float(np.mean(lead)),
            "median_lead_horizon_steps": float(np.median(lead)),
            "early_alert_rate": float(np.mean(lead > 0)),
            "on_time_alert_rate": float(np.mean(lead == 0)),
            "late_alert_rate": float(np.mean(lead < 0)),
        }
    else:
        lead_summary = {
            "n_event_cases": 0,
            "mean_lead_horizon_steps": 0.0,
            "median_lead_horizon_steps": 0.0,
            "early_alert_rate": 0.0,
            "on_time_alert_rate": 0.0,
            "late_alert_rate": 0.0,
        }

    summary = {
        "input": str(in_csv),
        "threshold_mw": float(threshold),
        "overall": overall,
        "lead_time_proxy": lead_summary,
    }

    out_json = out_dir / "hazard_utility_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {out_h_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    run(threshold=1.0)
