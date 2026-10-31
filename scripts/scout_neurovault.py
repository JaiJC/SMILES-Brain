#!/usr/bin/env python
"""Size the pool of open drug-response maps on NeuroVault by keyword, without ingesting anything.

    python scripts/scout_neurovault.py --out docs/results/neurovault_scouting.md

NeuroVault's `search` API parameter does not filter (it returns the whole database), so this
downloads the full collection list once (about 19 requests of 1000) and matches drug names in
each collection's name and description locally. A hit is **not** a usable validation map: each
candidate still needs manual review (drug-vs-placebo group contrast? unthresholded statistic?
registrable space?). Counts therefore overstate what is usable; the `placebo` column is a
cheap first filter, not a quality check.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

DRUGS = {
    "psilocybin": r"psilocyb", "LSD": r"\blsd\b|lysergic", "ketamine": r"ketamine",
    "MDMA": r"\bmdma\b|ecstasy", "methylphenidate": r"methylphenidate|ritalin",
    "amphetamine": r"amphetamine", "haloperidol": r"haloperidol", "risperidone": r"risperidone",
    "olanzapine": r"olanzapine", "SSRI (citalopram/fluoxetine/sertraline)":
        r"citalopram|fluoxetine|sertraline|paroxetine|\bssri",
    "benzodiazepine (midazolam/lorazepam/diazepam)": r"midazolam|lorazepam|diazepam|alprazolam",
    "propofol/anaesthetic": r"propofol|sevoflurane|dexmedetomidine",
    "opioid (naltrexone/morphine/fentanyl)": r"naltrexone|morphine|fentanyl|opioid",
    "oxytocin": r"oxytocin", "nicotine": r"nicotine", "cannabis/THC": r"\bthc\b|cannabi",
    "scopolamine/anticholinergic": r"scopolamine|biperiden",
    "levodopa/dopamine agonist": r"levodopa|l-dopa|bromocriptine|pramipexole|sulpiride",
    "caffeine": r"caffeine", "generic: placebo-controlled drug": r"placebo.{0,40}(drug|dose|mg)",
}
API = "https://neurovault.org/api/collections/?format=json&limit=1000&offset={}"
KEEP = ("id", "name", "description", "DOI", "number_of_images", "number_of_experimental_units",
        "group_comparison", "coordinate_space", "publication_status")


def download_all() -> list[dict]:
    out, offset = [], 0
    while True:
        req = urllib.request.Request(API.format(offset), headers={"User-Agent": "smilesbrain-scout/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https host
            page = json.load(resp)
        out += [{k: r.get(k) for k in KEEP} for r in page["results"]]
        print(f"  {len(out)}/{page['count']}")
        if not page.get("next") or not page["results"]:
            return out
        offset += 1000
        time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/neurovault_scouting.md"))
    ap.add_argument("--cache", type=Path, default=Path("artifacts/neurovault_collections.json"))
    args = ap.parse_args()
    if args.cache.exists():
        collections = json.loads(args.cache.read_text())
        print(f"using cached list ({len(collections)} collections)")
    else:
        collections = download_all()
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(collections))
    usable = [c for c in collections if (c.get("number_of_images") or 0) > 0
              and "temporary collection" not in (c.get("name") or "").lower()]
    placebo = re.compile(r"placebo|vehicle|saline", re.I)

    def stem(name: str) -> str:
        """Collapse auto-generated series ('... Meta Analysis: included : 2025-01-09 ...')."""
        return re.sub(r"[\s:_-]*\d[\d\s:._-]*$", "", (name or "").lower())[:50]

    table, hits = [], {}
    for label, pattern in DRUGS.items():
        rx = re.compile(pattern, re.I)
        found = [c for c in usable if rx.search(f"{c['name'] or ''} {c['description'] or ''}")]
        distinct = {stem(c["name"]) for c in found}
        with_placebo = [c for c in found if placebo.search(f"{c['name'] or ''} {c['description'] or ''}")]
        table.append((label, len(found), len(distinct), len({stem(c['name']) for c in with_placebo})))
        hits[label] = found
    filled = sum(c["number_of_experimental_units"] is not None for c in usable)
    md = ["# NeuroVault scouting: open collections that mention a drug", "",
          f"{len(collections)} collections downloaded; {len(usable)} non-empty, non-temporary. "
          "Matches are on name + description text only (image-level metadata was not searched).", "",
          "**A hit is not a usable validation map.** Each candidate needs manual review for a "
          "drug-vs-placebo group contrast, an unthresholded statistic, a registrable space, and "
          "(for this project) a readout comparable to resting CBF. Most hits below are task-fMRI "
          "or meta-analyses. Sample size cannot be filtered: `number_of_experimental_units` is "
          f"filled for only {filled} of {len(usable)} collections.", "",
          "'Distinct' collapses auto-generated series (e.g. one nicotine meta-analysis appears "
          "as hundreds of near-identical collections).", "",
          "| drug / class | collections matching | distinct | distinct, with placebo wording |",
          "|---|---:|---:|---:|", *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in table], "",
          "## Every collection matching a specific drug (not classes), for manual review", "",
          "| drug | id | images | name | DOI |", "|---|---:|---:|---|---|"]
    specific = ["psilocybin", "LSD", "ketamine", "MDMA", "methylphenidate", "amphetamine",
                "haloperidol", "risperidone", "olanzapine"]
    for drug in specific:
        seen_stems = set()
        for c in sorted(hits[drug], key=lambda c: -(c["number_of_images"] or 0)):
            if stem(c["name"]) in seen_stems:
                continue
            seen_stems.add(stem(c["name"]))
            name = (c["name"] or "").replace("|", "/")[:80]
            md.append(f"| {drug} | {c['id']} | {c['number_of_images']} | {name} | {c['DOI'] or ''} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    print("\n".join(md[:30]))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
