#!/usr/bin/env python
"""Plot a per-region score CSV (as written by scripts/predict.py) on the cortical surface.

    python scripts/plot_map.py artifacts/prediction.csv --out artifacts/prediction.png --html

Needs nilearn + matplotlib (`pip install -e '.[neuro,viz]'`); the Schaefer atlas and fsaverage5
surface are fetched by nilearn on first use. If that fails, a ranked-region chart labelled as a
fallback is written instead (exit status 0, message on stderr).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from smilesbrain.config import load_receptors
from smilesbrain.viz.surface import render_fallback, render_surface


def load_scores(path: Path, n_regions: int):
    df = pd.read_csv(path)
    if "score" not in df.columns:
        raise SystemExit(f"{path} needs a 'score' column (got {list(df.columns)})")
    if "region" in df.columns and not (df["region"].to_numpy() == range(len(df))).all():
        raise SystemExit("'region' column must be 0..N-1 in order; refusing to guess alignment")
    if len(df) != n_regions:
        raise SystemExit(f"{path} has {len(df)} regions; parcellation expects {n_regions}")
    return df["score"].to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", type=Path)
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: <scores>.png")
    ap.add_argument("--html", action="store_true", help="also write an interactive <out>.html")
    ap.add_argument("--title", default="")
    ap.add_argument("--data-dir", default=None, help="nilearn data cache dir")
    args = ap.parse_args()

    parc = load_receptors(parcellation=args.parcellation).parcellation
    if parc.subcortex or not parc.name.startswith("Schaefer"):
        raise SystemExit(f"{parc.key}: surface rendering supports cortical Schaefer parcellations")
    scores = load_scores(args.scores, parc.n_regions)
    out = args.out or args.scores.with_suffix(".png")
    try:
        render_surface(scores, out, title=args.title, data_dir=args.data_dir,
                       html=out.with_suffix(".html") if args.html else None)
        print(f"surface map -> {out}")
    except Exception as exc:  # missing extra or no network: degrade loudly, never silently
        print(f"WARNING: surface render failed ({type(exc).__name__}: {exc}); "
              "writing labelled fallback chart.", file=sys.stderr)
        render_fallback(scores, out, title=args.title, reason=type(exc).__name__)
        print(f"fallback chart -> {out}")


if __name__ == "__main__":
    main()
