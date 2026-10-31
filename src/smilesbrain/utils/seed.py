"""Reproducibility helpers for scripts and offline demos."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when the optional dependency is present."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(value: str, seed: int = 0) -> int:
    """Return a process-independent 32-bit seed derived from a string."""
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_manifest(
    output: str | Path,
    *,
    config: Any,
    seed: int,
    input_paths: Iterable[str | Path] = (),
) -> Path:
    """Write reproducibility metadata beside an output and return its path."""
    output_path = Path(output)
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    inputs = {}
    for raw_path in input_paths:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            inputs[str(path)] = _sha256(path)
    payload = {
        "config": config,
        "git_commit": _git_commit(),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": inputs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path
