"""
Revision helper: compare uncertainty methods in one table/plot.

Reads:
- outputs/deep_ensemble_multiresolution/uncertainty_metrics.csv (required)
- outputs/mc_dropout/uncertainty_metrics.csv (optional)
- outputs/bayesian/uncertainty_metrics.csv (optional)

Writes:
- outputs/revision_analysis/uncertainty_method_comparison.csv
- outputs/revision_analysis/uncertainty_method_comparison.png
- outputs/revision_analysis/uncertainty_baseline_template.csv (if optional files are missing)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_method(csv_path: Path, method_name: str, col_map: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if col_map:
        df = df.rename(columns=col_map)

    required = ["resolution", "coverage", "sharpness", "uncertainty_error_corr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    out = df[required].copy()
    out["method"] = method_name
    return out


def run() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "revision_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    de_csv = root / "outputs" / "deep_ensemble_multiresolution" / "uncertainty_metrics.csv"
    mc_csv = root / "outputs" / "mc_dropout" / "uncertainty_metrics.csv"
    bnn_csv = root / "outputs" / "bayesian" / "uncertainty_metrics.csv"

    if not de_csv.exists():
        raise FileNotFoundError(f"Required file not found: {de_csv}")

    frames = [
        load_method(
            de_csv,
            "Deep Ensemble",
            col_map={"uncertainty_error_corr": "uncertainty_error_corr", "resolution": "resolution"},
        )
    ]

    missing_optional = []

    if mc_csv.exists():
        frames.append(load_method(mc_csv, "MC Dropout"))
    else:
        missing_optional.append(str(mc_csv))

    if bnn_csv.exists():
        frames.append(load_method(bnn_csv, "Bayesian"))
    else:
        missing_optional.append(str(bnn_csv))

    comp = pd.concat(frames, ignore_index=True)

    out_csv = out_dir / "uncertainty_method_comparison.csv"
    comp.to_csv(out_csv, index=False)

    # Plot: coverage vs sharpness (lower sharpness is better, coverage near 95 is target)
    plt.figure(figsize=(8, 5))
    for method, g in comp.groupby("method"):
        plt.scatter(g["sharpness"], g["coverage"], label=method, s=70, alpha=0.85)

    plt.axhline(95.0, color="gray", linestyle="--", linewidth=1.0, label="Target 95% coverage")
    plt.xlabel("Sharpness (CI width)")
    plt.ylabel("Coverage (%)")
    plt.title("Uncertainty Method Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / "uncertainty_method_comparison.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    # If baseline files missing, emit template for easy filling
    if missing_optional:
        tpl = pd.DataFrame(
            {
                "resolution": ["1h", "2h", "4h", "6h", "12h", "24h"],
                "coverage": [None] * 6,
                "sharpness": [None] * 6,
                "uncertainty_error_corr": [None] * 6,
                "method": ["MC Dropout"] * 6,
            }
        )
        out_tpl = out_dir / "uncertainty_baseline_template.csv"
        tpl.to_csv(out_tpl, index=False)
        print("Optional baseline files not found:")
        for p in missing_optional:
            print(f"  - {p}")
        print(f"Template generated: {out_tpl}")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    run()
