"""Optional Chemprop D-MPNN adapter.

Chemprop is intentionally not a core dependency. This module validates the optional
dependency and provides the stable model boundary used by future GPU training jobs; it
does not silently fall back to fingerprints or an unrelated neural network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _require_chemprop() -> str:
    try:
        import chemprop
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Chemprop is required for the D-MPNN backend. Install a Chemprop release "
            "compatible with this environment before using backend: dmpnn."
        ) from exc
    if not hasattr(chemprop, "__version__") or not str(chemprop.__version__).startswith("1.7"):
        raise RuntimeError(
            "The D-MPNN adapter targets Chemprop 1.7.x. Install chemprop==1.7.1 "
            "or update this adapter for the installed major version."
        )
    return str(chemprop.__version__)


def _chemprop_command(name: str) -> str:
    """Resolve console scripts when the caller's PATH differs from its interpreter."""
    sibling = Path(sys.executable).with_name(name)
    return str(sibling) if sibling.exists() else name


@dataclass
class MaskedDMPNNRegressor:
    """Chemprop 1.7.x D-MPNN wrapper using its supported command-line interface."""

    receptor_keys: list[str]
    device: str = "cpu"
    epochs: int = 30
    ensemble_size: int = 1
    seed: int = 0
    checkpoint_path: Path | None = None
    checkpoint_paths: list[Path] | None = None

    def fit(self, smiles: list[str], Y: np.ndarray) -> MaskedDMPNNRegressor:
        _require_chemprop()
        if Y.shape != (len(smiles), len(self.receptor_keys)):
            raise ValueError("SMILES and target matrix dimensions do not match")
        empty = [key for key, count in zip(self.receptor_keys, np.isfinite(Y).sum(axis=0)) if count < 2]
        if empty:
            raise ValueError("D-MPNN targets need at least two observations: " + ", ".join(empty))
        work_path = Path(tempfile.mkdtemp(prefix="smilesbrain-dmpnn-"))
        data_path = work_path / "train.csv"
        frame = pd.DataFrame(Y, columns=self.receptor_keys)
        frame.insert(0, "smiles", smiles)
        frame.to_csv(data_path, index=False)
        save_dir = work_path / "run"
        command = [
            _chemprop_command("chemprop_train"), "--data_path", str(data_path),
            "--dataset_type", "regression", "--split_type", "random",
            "--split_sizes", "0.8", "0.1", "0.1", "--epochs", str(self.epochs),
            "--ensemble_size", str(self.ensemble_size), "--seed", str(self.seed),
            "--pytorch_seed", str(self.seed), "--save_dir", str(save_dir), "--quiet",
        ]
        if self.device == "cpu":
            command.append("--no_cuda")
        env = os.environ.copy()
        # Chemprop 1.7 checkpoints contain argparse.Namespace objects. PyTorch 2.6+
        # defaults to weights_only=True, which is incompatible with this trusted
        # local checkpoint format.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        subprocess.run(command, check=True, env=env)
        checkpoints = sorted(save_dir.rglob("model.pt"))
        if not checkpoints:
            raise RuntimeError(f"Chemprop completed without a model checkpoint in {save_dir}")
        self.checkpoint_paths = checkpoints
        self.checkpoint_path = checkpoints[0]
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        _require_chemprop()
        values, _ = self.predict_with_uncertainty(smiles)
        return values

    def predict_with_uncertainty(
        self, smiles: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        _require_chemprop()
        paths = self.checkpoint_paths or ([self.checkpoint_path] if self.checkpoint_path else [])
        if not paths or not all(path.exists() for path in paths):
            raise RuntimeError("D-MPNN has not been fitted or its checkpoint is missing")
        model_values = []
        with tempfile.TemporaryDirectory(prefix="smilesbrain-dmpnn-predict-") as work:
            test_path = Path(work) / "test.csv"
            pd.DataFrame({"smiles": smiles}).to_csv(test_path, index=False)
            for index, checkpoint in enumerate(paths):
                pred_path = Path(work) / f"predictions-{index}.csv"
                command = [
                    _chemprop_command("chemprop_predict"), "--test_path", str(test_path),
                    "--preds_path", str(pred_path), "--checkpoint_path", str(checkpoint),
                    "--no_cuda",
                ]
                if self.device != "cpu":
                    command.remove("--no_cuda")
                    command += ["--gpu", self.device]
                env = os.environ.copy()
                env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
                subprocess.run(command, check=True, env=env)
                model_values.append(
                    pd.read_csv(pred_path).select_dtypes(include=[np.number]).to_numpy()
                )
        values = np.stack(model_values)
        if values.shape[1:] != (len(smiles), len(self.receptor_keys)):
            raise RuntimeError(
                f"Chemprop returned {values.shape[1:]}, expected "
                f"({len(smiles)}, {len(self.receptor_keys)})"
            )
        return values.mean(axis=0), values.std(axis=0, ddof=1 if len(values) > 1 else 0)

    def save(self, path: str | Path) -> None:
        if self.checkpoint_path is None:
            raise RuntimeError("Cannot save an unfitted D-MPNN")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.checkpoint_path, destination)
        self.checkpoint_path = destination
        copied = [destination]
        for index, checkpoint in enumerate((self.checkpoint_paths or [])[1:], start=1):
            sibling = destination.with_name(f"{destination.stem}.ensemble-{index}{destination.suffix}")
            shutil.copy2(checkpoint, sibling)
            copied.append(sibling)
        self.checkpoint_paths = copied

    @classmethod
    def load(cls, path: str | Path, receptor_keys: list[str], device: str = "cpu"):
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        siblings = sorted(checkpoint.parent.glob(f"{checkpoint.stem}.ensemble-*{checkpoint.suffix}"))
        return cls(
            receptor_keys=receptor_keys,
            device=device,
            checkpoint_path=checkpoint,
            checkpoint_paths=[checkpoint, *siblings],
        )
