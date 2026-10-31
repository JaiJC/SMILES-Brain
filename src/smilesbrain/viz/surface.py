"""Render a region-score vector on the cortical surface (Schaefer parcellations).

Parcel labels are carried onto the fsaverage5 surface by sampling the *volumetric* Schaefer
atlas (the same label ids 1..N that `smilesbrain.data.parcellate` uses, so region i of any
vector in this project is parcel i + 1) with nearest-most-frequent interpolation. No
annotation files or extra downloads beyond nilearn's own are needed.

If nilearn is unavailable or its data cannot be fetched, `render_fallback` draws a ranked
region chart instead and says so on the figure; it is never passed off as a surface render.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEMIS = ("left", "right")
VIEWS = (("left", "lateral"), ("left", "medial"), ("right", "lateral"), ("right", "medial"))


def labels_to_vertex_scores(vertex_labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map per-vertex parcel ids (1..N, 0 = no parcel) to that parcel's score (NaN if none)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.rint(np.asarray(vertex_labels)).astype(int)
    out = np.full(labels.shape, np.nan)
    valid = (labels >= 1) & (labels <= len(scores))
    out[valid] = scores[labels[valid] - 1]
    return out


def symmetric_limit(scores: np.ndarray) -> float:
    """Colour limit that centres a diverging map on zero (1.0 for an all-zero/empty map)."""
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    peak = float(np.abs(finite).max()) if finite.size else 0.0
    return peak if peak > 0 else 1.0


def surface_vertex_labels(n_regions: int, *, mesh: str = "fsaverage5",
                          data_dir: str | None = None) -> dict[str, np.ndarray]:
    """Per-hemisphere vertex parcel ids (pial surface). Needs nilearn and its atlas data."""
    from nilearn import datasets, surface

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_regions, yeo_networks=7, resolution_mm=2, data_dir=data_dir
    )
    fsavg = datasets.fetch_surf_fsaverage(mesh, data_dir=data_dir)
    return {
        hemi: np.rint(
            surface.vol_to_surf(
                atlas.maps, fsavg[f"pial_{hemi}"], interpolation="nearest_most_frequent"
            )
        ).astype(int)
        for hemi in HEMIS
    }


def interactive_surface_html(scores: np.ndarray, *, mesh: str = "fsaverage5",
                             data_dir: str | None = None, width: int = 520,
                             height: int = 420) -> str:
    """Self-contained HTML with one rotatable 3D surface per hemisphere (nilearn view_surf)."""
    from nilearn import datasets, plotting

    scores = np.asarray(scores, dtype=float)
    vertex_labels = surface_vertex_labels(len(scores), mesh=mesh, data_dir=data_dir)
    fsavg = datasets.fetch_surf_fsaverage(mesh, data_dir=data_dir)
    limit = symmetric_limit(scores)
    frames = []
    for hemi in HEMIS:
        vertex_scores = np.nan_to_num(labels_to_vertex_scores(vertex_labels[hemi], scores))
        view = plotting.view_surf(
            fsavg[f"infl_{hemi}"], vertex_scores, bg_map=fsavg[f"sulc_{hemi}"],
            cmap="RdBu_r", vmax=limit, symmetric_cmap=True, title=f"{hemi} hemisphere",
        )
        frames.append(view.get_iframe(width=width, height=height))
    return ("<p><b>Hypothesis only:</b> affinity-weighted receptor density, "
            "not a validated drug map.</p>" + "".join(frames))


def render_surface(scores: np.ndarray, out: str | Path, *, title: str = "",
                   mesh: str = "fsaverage5", data_dir: str | None = None,
                   html: str | Path | None = None) -> Path:
    """Write a 4-view cortical-surface PNG (and optionally an interactive HTML) of `scores`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    scores = np.asarray(scores, dtype=float)
    vertex_labels = surface_vertex_labels(len(scores), mesh=mesh, data_dir=data_dir)
    fsavg = datasets.fetch_surf_fsaverage(mesh, data_dir=data_dir)
    vertex_scores = {h: labels_to_vertex_scores(vertex_labels[h], scores) for h in HEMIS}
    limit = symmetric_limit(scores)

    fig = plt.figure(figsize=(11, 6.4))
    for i, (hemi, view) in enumerate(VIEWS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        plotting.plot_surf_stat_map(
            fsavg[f"infl_{hemi}"], np.nan_to_num(vertex_scores[hemi]),
            hemi=hemi, view=view, bg_map=fsavg[f"sulc_{hemi}"], axes=ax,
            cmap="RdBu_r", vmin=-limit, vmax=limit, colorbar=False, figure=fig,
        )
        ax.set_title(f"{hemi} {view}", fontsize=9)
    mappable = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=matplotlib.colors.Normalize(-limit, limit))
    fig.colorbar(mappable, ax=fig.axes, shrink=0.5, label="predicted region score")
    fig.suptitle((title + "\n" if title else "")
                 + "Hypothesis only: affinity-weighted receptor density, not a validated drug map",
                 fontsize=10)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if html is not None:
        html = Path(html)
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(interactive_surface_html(scores, mesh=mesh, data_dir=data_dir))
    return out


def render_fallback(scores: np.ndarray, out: str | Path, *, title: str = "", top: int = 10,
                    reason: str = "") -> Path:
    """Ranked region chart used when no surface can be drawn. Clearly labelled as such."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float)
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))
    chosen = np.concatenate([order[:top], order[-top:]]) if len(order) > 2 * top else order
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(chosen) + 1.8))
    ax.barh([f"region {int(r)}" for r in chosen], scores[chosen],
            color=["#2a78d6" if scores[r] >= 0 else "#eb6834" for r in chosen])
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("predicted region score")
    ax.set_title(
        (title + "\n" if title else "")
        + "FALLBACK ranked chart, not a surface render" + (f" ({reason})" if reason else ""),
        fontsize=9)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
