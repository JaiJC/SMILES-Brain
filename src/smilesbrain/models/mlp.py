"""Masked multi-task MLP affinity model (PyTorch, GPU-ready for SLURM).

Shares a feature trunk across all 19 receptors and predicts a pChEMBL per receptor. The loss
is masked: molecules with no label for a receptor contribute no gradient to that head, so the
whole sparse ChEMBL table trains one model and the shared trunk learns chemistry that
transfers across receptors (the point of multi-task learning here).

Torch is an optional dependency; importing this module without torch raises a clear message
only when the class is instantiated, keeping the rest of the package import-light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "PyTorch is required for the MLP model. Install with `pip install -e '.[torch]'`."
        ) from exc


@dataclass
class MLPConfig:
    hidden: tuple[int, ...] = (512, 256)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 256
    device: str = "cpu"


class MaskedMultiTaskMLP:
    def __init__(self, n_features: int, n_receptors: int, cfg: MLPConfig | None = None):
        torch, nn = _require_torch()
        self.torch, self.nn = torch, nn
        self.cfg = cfg or MLPConfig()
        self.n_receptors = n_receptors
        layers: list = []
        dim = n_features
        for h in self.cfg.hidden:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(self.cfg.dropout)]
            dim = h
        layers += [nn.Linear(dim, n_receptors)]
        self.net = nn.Sequential(*layers).to(self.cfg.device)

    def _masked_mse(self, pred, target, mask):
        diff = (pred - target) * mask
        denom = mask.sum().clamp_min(1.0)
        return (diff ** 2).sum() / denom

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None, verbose=True):
        torch = self.torch
        dev = self.cfg.device
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        Yt = torch.as_tensor(np.nan_to_num(Y, nan=0.0), dtype=torch.float32, device=dev)
        Mt = torch.as_tensor(~np.isnan(Y), dtype=torch.float32, device=dev)

        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        n = Xt.shape[0]
        for epoch in range(self.cfg.epochs):
            self.net.train()
            perm = torch.randperm(n, device=dev)
            total = 0.0
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                opt.zero_grad()
                loss = self._masked_mse(self.net(Xt[idx]), Yt[idx], Mt[idx])
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            if verbose and (epoch % 10 == 0 or epoch == self.cfg.epochs - 1):
                msg = f"epoch {epoch:4d}  train_mse {total / n:.4f}"
                if X_val is not None and Y_val is not None:
                    msg += f"  val_mse {self.score(X_val, Y_val):.4f}"
                print(msg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32, device=self.cfg.device)
            return self.net(Xt).cpu().numpy()

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Masked MSE on observed entries."""
        pred = self.predict(X)
        mask = ~np.isnan(Y)
        diff = (pred - np.nan_to_num(Y)) * mask
        return float((diff ** 2).sum() / max(mask.sum(), 1))

    def save(self, path: str) -> None:
        self.torch.save({"state_dict": self.net.state_dict(),
                         "n_receptors": self.n_receptors}, path)
