"""Turn a volumetric group-level statistical map (NIfTI) into one value per atlas region.

The receptor atlas, the observed drug maps and the spin nulls must all share one parcellation,
and the region order must match. For the Schaefer parcellations the volumetric labels are
1..N with left hemisphere first, which is the row order of the receptor atlas, so the value
for label k lands in row k-1.

nilearn's `NiftiLabelsMasker` does the averaging (resampling the labels to the map's grid with
nearest-neighbour) and drops labels that contain no voxels. It silently turns NaN voxels into
0, which would drag a region's mean toward zero, so we run it on the NaN-zeroed map and on a
finite-voxel indicator and divide: that is exactly the NaN-ignoring mean. We put those
back as NaN rather than shifting the vector, so a partially-covered map keeps its alignment;
the harness's correlation already ignores NaN regions, and the count is recorded in the
provenance sidecar so it is never silent.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def fetch_schaefer_labels_img(n_regions: int, resolution_mm: int = 2, data_dir: str | None = None):
    """Schaefer-2018 (7 network) volumetric label image in MNI space, via nilearn."""
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_regions, yeo_networks=7, resolution_mm=resolution_mm, data_dir=data_dir
    )
    return atlas["maps"]


def parcellate_nifti(
    stat_img, labels_img, n_regions: int, zero_as_missing: bool = False
) -> np.ndarray:
    """Average `stat_img` within each label of `labels_img` -> (n_regions,) vector.

    Labels must be the integers 1..n_regions (background 0). Regions with no finite voxels in
    the statistical map come back as NaN. Always the mean over finite voxels.

    `zero_as_missing`: some deposited maps are masked with exact 0 outside the analysis mask
    rather than NaN. Averaging those zeros in would shrink edge regions toward 0, so set this
    to treat exact zeros as "no data". It is opt-in because 0 is also a legitimate value.
    """
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker

    stat_img = nib.load(stat_img) if isinstance(stat_img, (str, Path)) else stat_img
    labels_img = nib.load(labels_img) if isinstance(labels_img, (str, Path)) else labels_img

    present = np.unique(np.asarray(labels_img.dataobj).astype(int))
    expected = set(range(1, n_regions + 1))
    unexpected = sorted(set(present) - expected - {0})
    if unexpected:
        raise ValueError(
            f"label image has ids outside 1..{n_regions} (e.g. {unexpected[:5]}); "
            "it is not the configured parcellation."
        )

    data = np.asarray(stat_img.dataobj, dtype=float)
    if data.ndim > 3:
        # SPM/FSL often write a single volume as 4D with a trailing singleton axis; that is
        # still one map. More than one real volume is ambiguous, so refuse it.
        extra = data.shape[3:]
        if any(d != 1 for d in extra):
            raise ValueError(
                f"expected a single statistical map, got {extra[0]} volumes (shape {data.shape}); "
                "split it and ingest one contrast at a time."
            )
        data = data.reshape(data.shape[:3])
        stat_img = nib.Nifti1Image(data, stat_img.affine)
    if data.ndim != 3:
        raise ValueError(f"expected a single 3D statistical map, got shape {data.shape}")
    finite = np.isfinite(data) & ((data != 0) if zero_as_missing else True)
    zeroed = nib.Nifti1Image(np.where(finite, data, 0.0).astype("float32"), stat_img.affine)
    indicator = nib.Nifti1Image(finite.astype("float32"), stat_img.affine)

    masker = NiftiLabelsMasker(labels_img, strategy="mean", resampling_target="data")
    mean_zeroed = np.asarray(masker.fit_transform(zeroed), dtype=float).reshape(-1)
    frac_finite = np.asarray(masker.transform(indicator), dtype=float).reshape(-1)

    ids = [v for k, v in masker.region_ids_.items() if k != "background"]
    out = np.full(n_regions, np.nan)
    for col, label_id in enumerate(ids):
        if frac_finite[col] > 0:
            out[int(label_id) - 1] = mean_zeroed[col] / frac_finite[col]
    return out


def write_observed_map(values: np.ndarray, path: str | Path, n_regions: int) -> Path:
    """Write the one-value-per-region CSV the harness's `load_observed_map` reads."""
    from smilesbrain.validation.maps import load_observed_map

    path = Path(path)
    if values.shape != (n_regions,):
        raise ValueError(f"expected {n_regions} values, got shape {values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=",", fmt="%.8g")
    back = load_observed_map(path, n_regions)  # round-trip check: fails loudly on mismatch
    np.testing.assert_allclose(back, values, rtol=1e-6, equal_nan=True)
    return path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(
    csv_path: str | Path,
    values: np.ndarray,
    *,
    drug: str,
    source: str,
    n_subjects: int | None,
    contrast: str,
    parcellation: str,
    input_nifti: str | Path,
    measure: str = "",
    zero_as_missing: bool = False,
    extra: dict | None = None,
) -> Path:
    """Sidecar `<csv>.provenance.json`: where the map came from and how it was parcellated."""
    csv_path = Path(csv_path)
    side = csv_path.with_name(csv_path.name + ".provenance.json")
    record = {
        "drug": drug,
        "source": source,
        "n_subjects": n_subjects,
        "contrast": contrast,
        "measure": measure,
        "parcellation": parcellation,
        "strategy": "mean over finite voxels",
        "zero_as_missing": zero_as_missing,
        "n_regions": int(values.shape[0]),
        "n_regions_missing": int(np.isnan(values).sum()),
        "input_nifti": str(input_nifti),
        "input_sha256": sha256(input_nifti),
        "output_csv": str(csv_path),
        "output_sha256": sha256(csv_path),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    with open(side, "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return side
