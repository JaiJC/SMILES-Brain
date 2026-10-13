"""Download and normalize the public PDSP Ki database.

PDSP publishes receptor names rather than the UniProt accession used by our config.
The receptor-name aliases below are deliberately explicit so target matching is
auditable instead of silently accepting broad substring matches.
"""

from __future__ import annotations

import argparse
import io
import math
import re
from pathlib import Path

import pandas as pd
import requests

from smilesbrain.config import ReceptorConfig, load_receptors

PDSP_URL = "https://pdsp.med.unc.edu/databases/kiDownload/download.php"
TARGET_ALIASES = {
    "5HT1a": ("5-HT1A", "5HT1A", "SEROTONIN 1A"),
    "5HT1b": ("5-HT1B", "5HT1B", "SEROTONIN 1B"),
    "5HT2a": ("5-HT2A", "5HT2A", "SEROTONIN 2A"),
    "5HT4": ("5-HT4", "5HT4", "SEROTONIN 4"),
    "5HT6": ("5-HT6", "5HT6", "SEROTONIN 6"),
    "5HTT": ("SERT", "5-HT TRANSPORTER", "SEROTONIN TRANSPORTER"),
    "A4B2": ("ALPHA4BETA2", "A4B2", "ALPHA 4 BETA 2"),
    "CB1": ("CB1", "CANNABINOID 1"),
    "D1": ("D1", "DOPAMINE D1"),
    "D2": ("D2", "DOPAMINE D2"),
    "DAT": ("DAT", "DOPAMINE TRANSPORTER"),
    "GABAa": ("GABAA", "GABA-A", "BENZODIAZEPINE"),
    "H3": ("H3", "HISTAMINE H3"),
    "M1": ("M1", "MUSCARINIC M1"),
    "mGluR5": ("MGLUR5", "MGLU5"),
    "MU": ("MU", "MOR", "MU OPIOID"),
    "NAT": ("NET", "NAT", "NOREPINEPHRINE TRANSPORTER"),
    "NMDA": ("NMDA", "NMDAR", "GLUN2B"),
    "VAChT": ("VACHT", "VESICULAR ACETYLCHOLINE"),
}


def _target_matches(value: object, aliases: tuple[str, ...]) -> bool:
    target = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return any(re.sub(r"[^A-Z0-9]", "", alias.upper()) in target for alias in aliases)


def download_database(url: str = PDSP_URL) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(response.content),
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(
        "PDSP download is not readable as UTF-8, Windows-1252, or Latin-1"
    ) from last_error


def _column(frame: pd.DataFrame, *names: str) -> str:
    normalized = {str(c).strip().lower().replace(" ", ""): c for c in frame.columns}
    for name in names:
        if name.lower().replace(" ", "") in normalized:
            return normalized[name.lower().replace(" ", "")]
    raise ValueError(f"PDSP file is missing one of columns: {names}")


def ki_to_pki(value: object, note: object = "") -> float | None:
    """Convert a PDSP Ki value to pKi, handling common textual units."""
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if not match:
        return None
    number = float(match.group())
    if number <= 0:
        return None
    text = f"{value} {note}".lower()
    if "pm" in text:
        molar = number * 1e-12
    elif "um" in text or "µm" in text:
        molar = number * 1e-6
    elif re.search(r"\bmm\b", text):
        molar = number * 1e-3
    elif re.search(r"\bmol(?:ar)?\b", text):
        molar = number
    else:
        molar = number * 1e-9  # PDSP Ki values are conventionally nM.
    return float(-math.log10(molar))


def normalize_pdsp(frame: pd.DataFrame, receptor_key: str) -> pd.DataFrame:
    receptor_col = _column(frame, "Receptor", "name")
    smiles_col = _column(frame, "SMILES", "smiles")
    value_col = _column(frame, "Ki value", "kival", "Ki")
    note_col = next(
        (
            c
            for c in frame.columns
            if str(c).strip().lower().replace(" ", "") == "kinote"
        ),
        None,
    )
    aliases = TARGET_ALIASES.get(receptor_key, ())
    target = frame[receptor_col].fillna("").astype(str).str.upper()
    mask = target.map(lambda value: _target_matches(value, aliases))
    selected = frame.loc[mask].copy()
    selected["pchembl"] = [
        ki_to_pki(value, selected.iloc[i][note_col] if note_col else "")
        for i, value in enumerate(selected[value_col])
    ]
    selected = selected.rename(columns={smiles_col: "canonical_smiles"})
    selected["source"] = "pdsp"
    return selected[["canonical_smiles", "pchembl", "source"]].dropna(
        subset=["canonical_smiles", "pchembl"]
    )


def fetch_all(out_dir: Path, config: ReceptorConfig, url: str = PDSP_URL) -> None:
    raw = download_database(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    for receptor in config.receptors:
        normalized = normalize_pdsp(raw, receptor.atlas_key)
        normalized.to_csv(out_dir / f"{receptor.atlas_key}.csv", index=False)
        print(f"{receptor.atlas_key:8s}: {len(normalized):6d} PDSP molecules")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PDSP Ki affinities.")
    parser.add_argument("--out-dir", default="data/affinity_pdsp", type=Path)
    parser.add_argument("--url", default=PDSP_URL)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    fetch_all(args.out_dir, load_receptors(args.config), args.url)


if __name__ == "__main__":
    main()
