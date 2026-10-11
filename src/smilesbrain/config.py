"""Loading and validation of the receptor configuration.

The receptor list is the single source of truth that ties together the three pipeline
stages: the affinity model predicts one value per receptor, the projection step expects a
density matrix whose columns are these same receptors in the same order, and validation
reports per-receptor. Keeping the order canonical here prevents silent column misalignment
-- the most dangerous and least visible bug in this kind of pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "receptors.yaml"


@dataclass(frozen=True)
class Receptor:
    atlas_key: str
    name: str
    uniprot: str
    system: str
    complex: bool = False
    sparse: bool = False


@dataclass(frozen=True)
class ReceptorConfig:
    receptors: tuple[Receptor, ...]
    parcellation_name: str
    n_regions: int

    @property
    def atlas_keys(self) -> list[str]:
        """Canonical receptor order. Every matrix column must follow this."""
        return [r.atlas_key for r in self.receptors]

    @property
    def uniprots(self) -> list[str]:
        return [r.uniprot for r in self.receptors]

    @property
    def n_receptors(self) -> int:
        return len(self.receptors)

    def index_of(self, atlas_key: str) -> int:
        return self.atlas_keys.index(atlas_key)


def load_receptors(path: str | Path | None = None) -> ReceptorConfig:
    path = Path(path) if path is not None else _DEFAULT_CONFIG
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    receptors = tuple(
        Receptor(
            atlas_key=str(r["atlas_key"]),
            name=str(r["name"]),
            uniprot=str(r["uniprot"]),
            system=str(r["system"]),
            complex=bool(r.get("complex", False)),
            sparse=bool(r.get("sparse", False)),
        )
        for r in raw["receptors"]
    )

    keys = [r.atlas_key for r in receptors]
    if len(keys) != len(set(keys)):
        dupes = {k for k in keys if keys.count(k) > 1}
        raise ValueError(f"Duplicate atlas_key(s) in receptor config: {sorted(dupes)}")

    parc = raw["parcellation"]
    return ReceptorConfig(
        receptors=receptors,
        parcellation_name=str(parc["name"]),
        n_regions=int(parc["n_regions"]),
    )
