"""
Revision helper: paired Wilcoxon signed-rank tests for MR vs MH.

Reads:
    outputs/comparison_mr_mh/comparison_metrics.csv

Writes:
    outputs/revision_analysis/wilcoxon_significance.csv
    outputs/revision_analysis/wilcoxon_significance_summary.txt

Notes:
- Current input has one row per horizon (typically n=6).
- This script tests paired differences across horizons.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def rankdata_average(values: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return average ranks (1..n), handling ties by average rank."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(x)
    ranks = np.zeros(n, dtype=float)

    i = 0
    rank_start = 1
    while i < n:
        j = i
        while j + 1 < n and abs(x[order[j + 1]] - x[order[i]]) <= tol:
            j += 1

        block_size = j - i + 1
        avg_rank = (rank_start + (rank_start + block_size - 1)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank

        rank_start += block_size
        i = j + 1

    return ranks


def _exact_two_sided_p_from_ranks(ranks: np.ndarray, obs_w_plus: float) -> float:
    """Exact two-sided p-value via subset-sum DP over signed ranks."""
    # Ranks are integer/half-integer. Scale by 2 to use exact integer sums.
    scaled = np.rint(2.0 * np.asarray(ranks, dtype=float)).astype(int)
    obs = int(round(2.0 * obs_w_plus))

    counts = {0: 1}
    for r in scaled:
        new_counts = dict(counts)
        for s, c in counts.items():
            new_counts[s + int(r)] = new_counts.get(s + int(r), 0) + c
        counts = new_counts

    total = float(2 ** len(scaled))
    p_lower = sum(c for s, c in counts.items() if s <= obs) / total
    p_upper = sum(c for s, c in counts.items() if s >= obs) / total
    return min(1.0, 2.0 * min(p_lower, p_upper))


def wilcoxon_signed_rank_pvalue(diffs: np.ndarray) -> tuple[float, float, int, str]:
    """
    Two-sided Wilcoxon signed-rank p-value.

    Returns:
        (w_stat, p_value, n_nonzero, method)
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]

    # Wilcoxon excludes exact zeros.
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0, 0, "degenerate"

    ranks = rankdata_average(np.abs(nz))
    w_plus = float(np.sum(ranks[nz > 0]))
    w_minus = float(np.sum(ranks[nz < 0]))
    w_stat = min(w_plus, w_minus)

    # Exact method for small n, normal approximation otherwise.
    if n <= 20:
        p_value = _exact_two_sided_p_from_ranks(ranks, w_plus)
        method = "exact"
    else:
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
        z = max(0.0, z)
        p_value = math.erfc(z / math.sqrt(2.0))
        method = "normal_approx"

    return float(w_stat), float(p_value), int(n), method


def bootstrap_ci_mean(diffs: np.ndarray, n_boot: int = 20000, seed: int = 123) -> tuple[float, float, float]:
    """Bootstrap mean and 95% percentile CI."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
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
        clean = diffs[np.isfinite(diffs)]
        w_stat, w_p, n_nonzero, method = wilcoxon_signed_rank_pvalue(clean)
        mean_diff, ci_low, ci_high = bootstrap_ci_mean(clean)

        rows.append(
            {
                "metric": metric,
                "n_pairs": int(len(clean)),
                "n_nonzero": int(n_nonzero),
                "mean_paired_diff": mean_diff,
                "median_paired_diff": float(np.median(clean)) if len(clean) else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_w_stat": w_stat,
                "wilcoxon_pvalue": w_p,
                "wilcoxon_method": method,
                "interpretation": note,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = out_dir / "wilcoxon_significance.csv"
    out_df.to_csv(out_csv, index=False)

    summary = [
        "Paired Wilcoxon signed-rank summary for MR vs MH",
        f"Input: {in_csv}",
        f"Rows (horizons): {len(df)}",
        "",
        "Warning: with comparison_metrics.csv this is horizon-level testing, not per-sample testing.",
        "",
    ]
    for r in rows:
        summary.append(
            f"- {r['metric']}: mean diff={r['mean_paired_diff']:.4f} "
            f"(95% CI [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]), "
            f"W={r['wilcoxon_w_stat']:.4f}, p={r['wilcoxon_pvalue']:.4g} ({r['wilcoxon_method']})"
        )

    out_txt = out_dir / "wilcoxon_significance_summary.txt"
    out_txt.write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    run()
