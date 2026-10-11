"""Load the validation drug-map catalog (configs/validation_maps.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT = Path(__file__).resolve().parents[3] / "configs" / "validation_maps.yaml"


@dataclass(frozen=True)
class DrugMap:
    drug: str
    smiles: str
    measure: str
    file: str
    source: str
    expected_sign: str | None = None          # "positive" | "negative" | None
    measured_profile: str | None = None        # optional affinity CSV path

    @property
    def expected_sign_value(self) -> int | None:
        return {"positive": 1, "negative": -1}.get(self.expected_sign or "", None)


def load_catalog(path: str | Path | None = None) -> list[DrugMap]:
    path = Path(path) if path is not None else _DEFAULT
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    out: list[DrugMap] = []
    for m in raw["maps"]:
        out.append(
            DrugMap(
                drug=str(m["drug"]),
                smiles=str(m["smiles"]),
                measure=str(m.get("measure", "")),
                file=str(m["file"]),
                source=str(m.get("source", "")),
                expected_sign=m.get("expected_sign"),
                measured_profile=m.get("measured_profile"),
            )
        )
    return out
