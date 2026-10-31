"""Headline numbers and figures regenerated from saved artifacts (no model, no network).

`summarise_tiers` turns the per-tier table written by `scripts/evaluate.py --report`
(`artifacts/validation_tiers.csv`) into the counts quoted in the write-up, so the report's
numbers are computed, not typed. Nothing here re-runs an evaluation or invents a value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["tier", "drug", "r", "p", "spin", "n_perm", "sign_ok", "p_bh"]
ALPHA = 0.05


def check_tiers(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"tier table is missing columns {missing}; re-run evaluate.py --report")
    if df.empty:
        raise ValueError("tier table is empty")


def summarise_tiers(df: pd.DataFrame) -> dict[str, dict]:
    """Per tier: n drugs, r summary, significant counts (raw and BH), sign agreement, null type."""
    check_tiers(df)
    out: dict[str, dict] = {}
    for tier, g in df.groupby("tier", sort=True):
        signs = g["sign_ok"].dropna().astype(bool)
        out[str(tier)] = {
            "n_drugs": int(len(g)),
            "median_r": float(np.nanmedian(g["r"])),
            "mean_r": float(np.nanmean(g["r"])),
            "n_significant": int((g["p"] < ALPHA).sum()),
            "n_significant_bh": int((g["p_bh"] < ALPHA).sum()),
            "n_sign_testable": int(len(signs)),
            "n_sign_correct": int(signs.sum()),
            "all_spin_tests": bool(g["spin"].astype(bool).all()),
            "n_perm": int(g["n_perm"].max()),
        }
    return out


def summary_markdown(summary: dict[str, dict]) -> str:
    lines = [
        "| tier | n drugs | median r | p<0.05 | BH p<0.05 | expected sign correct | null |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for tier, s in summary.items():
        null = (f"spin ({s['n_perm']} rotations)" if s["all_spin_tests"]
                else "NAIVE (not a spin test)")
        n = s["n_drugs"]
        lines.append(
            f"| {tier} | {n} | {s['median_r']:+.3f} | {s['n_significant']}/{n}"
            f" | {s['n_significant_bh']}/{n}"
            f" | {s['n_sign_correct']}/{s['n_sign_testable']} | {null} |"
        )
    return "\n".join(lines)


def plot_tier_r(df: pd.DataFrame, path: str | Path) -> Path:
    """Dot plot of r per drug for tiers A and B; filled = spin p < 0.05."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    check_tiers(df)
    colours = {"A": "#2a78d6", "B": "#eb6834"}
    labels = {"A": "A: measured affinity", "B": "B: zero-shot (chemotype held out)"}
    sub = df[df["tier"].isin(colours)]
    drugs = list(dict.fromkeys(sub["drug"]))
    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(drugs) + 2), 4.2))
    for j, tier in enumerate(colours):
        g = sub[sub["tier"] == tier].set_index("drug").reindex(drugs)
        x = np.arange(len(drugs)) + (j - 0.5) * 0.28
        sig = (g["p"] < ALPHA).to_numpy()
        ax.scatter(x[sig], g["r"].to_numpy()[sig], s=60, color=colours[tier], zorder=3)
        ax.scatter(x[~sig], g["r"].to_numpy()[~sig], s=60, facecolors="none",
                   edgecolors=colours[tier], linewidths=1.6, zorder=3)
        ax.scatter([], [], s=60, color=colours[tier], label=labels[tier])
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(drugs)), drugs, rotation=35, ha="right")
    ax.set_ylabel("Pearson r (predicted vs observed map)")
    spin = bool(sub["spin"].astype(bool).all())
    ax.set_title("filled = spin p < 0.05; hollow = not significant"
                 + ("" if spin else "  |  NAIVE NULL, not a spin test"), fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_model_compare(
    df: pd.DataFrame, path: str | Path, column: str = "mean_r2_non_sparse"
) -> Path:
    """Bar chart of held-out R^2 per affinity backend (scaffold split)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "backend" not in df.columns or column not in df.columns:
        raise ValueError(f"model comparison needs columns 'backend' and '{column}'")
    latest = df.drop_duplicates("backend", keep="last")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(latest["backend"], latest[column], color="#2a78d6")
    for i, v in enumerate(latest[column]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"{column} (scaffold split)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
