"""
Revision helper: paired statistical significance tests for MR vs MH.

Reads:
    outputs/comparison_mr_mh/comparison_metrics.csv

Writes:
    outputs/revision_analysis/statistical_significance.csv
    outputs/revision_analysis/statistical_significance_summary.txt

This script is dependency-light (numpy/pandas only) and uses:
- paired sign test (exact binomial, two-sided)
- paired permutation sign-flip test (two-sided)
- bootstrap confidence interval for mean paired difference
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def two_sided_sign_test_pvalue(n_pos: int, n_nonzero: int) -> float:
    """Exact two-sided sign test p-value using Binomial(n, 0.5)."""
    if n_nonzero == 0:
        return 1.0
    k = min(n_pos, n_nonzero - n_pos)
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n_nonzero, i) * (0.5 ** n_nonzero)
    return min(1.0, 2.0 * p)


def permutation_sign_flip_pvalue(diffs: np.ndarray, n_perm: int = 20000, seed: int = 42) -> float:
    """Two-sided paired permutation test using random sign flips."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    obs = float(np.mean(diffs))
    if np.allclose(diffs, 0.0):
        return 1.0

    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm_means = np.mean(signs * diffs[None, :], axis=1)
    p = float(np.mean(np.abs(perm_means) >= abs(obs)))
    return max(p, 1.0 / n_perm)


def bootstrap_ci_mean(diffs: np.ndarray, n_boot: int = 20000, seed: int = 123) -> tuple[float, float, float]:
    """Bootstrap mean and 95% percentile CI."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = np.mean(diffs[idx], axis=1)
    return float(np.mean(diffs)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run() -> None:
    root = Path(__file__).resolve().parents[1]
    in_csv = root / "outputs" / "comparison_mr_mh" / "comparison_metrics.csv"
    out_dir = root / "outputs" / "revision_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input not found: {in_csv}")

    df = pd.read_csv(in_csv)

    tests = [
        ("rmse", (df["mh_rmse"] - df["mr_rmse"]).to_numpy(), "MR better when diff > 0"),
        ("mae", (df["mh_mae"] - df["mr_mae"]).to_numpy(), "MR better when diff > 0"),
        ("r2", (df["mr_r2"] - df["mh_r2"]).to_numpy(), "MR better when diff > 0"),
    ]

    rows = []
    for metric, diffs, note in tests:
        nz = diffs[diffs != 0]
        n_pos = int(np.sum(nz > 0))
        n_nonzero = int(len(nz))
        sign_p = two_sided_sign_test_pvalue(n_pos, n_nonzero)
        perm_p = permutation_sign_flip_pvalue(diffs)
        mean_diff, ci_low, ci_high = bootstrap_ci_mean(diffs)

        rows.append(
            {
                "metric": metric,
                "n_pairs": int(len(diffs)),
                "mean_paired_diff": mean_diff,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sign_test_pvalue": sign_p,
                "perm_test_pvalue": perm_p,
                "interpretation": note,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / "statistical_significance.csv"
    out_df.to_csv(out_csv, index=False)

    summary = [
        "Paired significance summary for MR vs MH",
        f"Input: {in_csv}",
        f"Rows (horizons): {len(df)}",
        "",
    ]
    for r in rows:
        summary.append(
            f"- {r['metric']}: mean diff={r['mean_paired_diff']:.4f} "
            f"(95% CI [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]), "
            f"sign p={r['sign_test_pvalue']:.4g}, perm p={r['perm_test_pvalue']:.4g}"
        )

    out_txt = out_dir / "statistical_significance_summary.txt"
    out_txt.write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    run()
