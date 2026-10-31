#!/usr/bin/env python
"""Regenerate every figure and headline number of the write-up from saved artifacts.

    python scripts/make_figures.py    # reads artifacts/, writes artifacts/figures/

Inputs (all produced by earlier steps; nothing is recomputed or invented here):
  artifacts/validation_tiers.csv   from  scripts/evaluate.py --report      (required)
  artifacts/model_compare.csv      from  scripts/train_affinity.py          (optional)

Outputs in --out-dir: fig_validation_r.png, fig_model_compare.png (if input exists),
numbers.json and numbers.md (the counts quoted in docs/REPORT.md), plus a manifest.
Fails loudly if the required input is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smilesbrain.utils import write_run_manifest
from smilesbrain.validation.summary import (
    plot_model_compare,
    plot_tier_r,
    summarise_tiers,
    summary_markdown,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=Path, default=Path("artifacts/validation_tiers.csv"))
    ap.add_argument("--model-compare", type=Path, default=Path("artifacts/model_compare.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/figures"))
    args = ap.parse_args()

    if not args.tiers.exists():
        raise SystemExit(
            f"{args.tiers} not found. Run `python scripts/evaluate.py --report` on real data "
            "first (docs/VALIDATION.md). Refusing to draw figures from nothing."
        )
    tiers = pd.read_csv(args.tiers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise_tiers(tiers)
    (args.out_dir / "numbers.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (args.out_dir / "numbers.md").write_text(summary_markdown(summary) + "\n")
    print(summary_markdown(summary))
    outputs = [plot_tier_r(tiers, args.out_dir / "fig_validation_r.png")]
    inputs = [args.tiers]
    if args.model_compare.exists():
        outputs.append(plot_model_compare(pd.read_csv(args.model_compare),
                                          args.out_dir / "fig_model_compare.png"))
        inputs.append(args.model_compare)
    else:
        print(f"note: {args.model_compare} not found; skipping the model-comparison figure.")
    manifest = write_run_manifest(args.out_dir / "numbers.json",
                                  config={"outputs": [str(o) for o in outputs]},
                                  seed=0, input_paths=inputs)
    print("\n".join(f"wrote {o}" for o in outputs), f"\nmanifest -> {manifest}", sep="\n")


if __name__ == "__main__":
    main()
