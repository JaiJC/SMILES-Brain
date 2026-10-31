"""Interactive demo: type a SMILES, see predicted receptor affinities and a brain map.

    streamlit run app/app.py

Uses the trained artifact and real atlas when they exist (paths configurable in the sidebar);
otherwise it falls back to a synthetic model/atlas and shows a DEMO banner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from smilesbrain.config import load_receptors
from smilesbrain.projection import parse_direction_spec
from smilesbrain.service import DEMO_BANNER, load_atlas, load_model, predict_molecule

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    "Haloperidol": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
    "Escitalopram": "CN(C)CCC[C@@]1(OCc2cc(C#N)ccc12)c1ccc(F)cc1",
    "Midazolam": "Cc1ncc2CN=C(c3ccccc3F)c3cc(Cl)ccc3-n12",
}

LIMITATIONS = """
**Read before trusting anything below.**
- On 8 drugs with open CBF maps, **binding affinity did not predict the observed brain maps
  better than randomly reassigning the same affinities among receptors** (0 of 6 testable
  drugs beat that null; see `docs/REPORT.md`). Receptor densities *can* explain the maps; the
  affinity-to-weight step is the open problem.
- Receptor density is where a drug *can* act, not where it *does* (no pharmacokinetics).
- Direction (agonist vs antagonist) is never predicted; by default every receptor is treated
  as if the molecule were an agonist.
- This is a hypothesis generator for prioritisation, not a clinical or preclinical claim.
"""


@st.cache_resource(show_spinner="Loading model and atlas...")
def _load(artifact: str, atlas: str, parcellation: str | None):
    rc = load_receptors(parcellation=parcellation)
    bundle, demo_model = load_model(Path(artifact), rc)
    atlas_df, demo_atlas = load_atlas(Path(atlas or rc.parcellation.atlas_file), rc)
    return rc, bundle, atlas_df, demo_model, demo_atlas


@st.cache_data(show_spinner="Rendering interactive surface...")
def _surface_html(scores: tuple[float, ...]) -> str | None:
    try:
        from smilesbrain.viz.surface import interactive_surface_html

        return interactive_surface_html(np.asarray(scores))
    except Exception as exc:  # nilearn missing / data unreachable
        st.session_state["surface_error"] = f"{type(exc).__name__}: {exc}"
        return None


def main() -> None:
    st.set_page_config(page_title="SMILES-Brain", layout="wide")
    st.title("SMILES → Brain")
    st.caption("Predict a molecule's receptor-affinity profile and project it on PET receptor "
               "density. **Unvalidated hypothesis generator.**")

    with st.sidebar:
        st.header("Settings")
        artifact = st.text_input("Model artifact", str(ROOT / "artifacts/affinity_baseline.joblib"))
        parcellation = st.text_input("Parcellation key (blank = default)", "") or None
        atlas_path = st.text_input("Atlas CSV (blank = parcellation default)", "")
        weight_mode = st.selectbox("Weight mode", ["zscore", "raw", "softmax"])
        direction_spec = st.text_input("Directions (optional)", "",
                                       help='e.g. "5HT2a=+1,D2=-1"; blank = magnitude-only')

    rc, bundle, atlas, demo_model, demo_atlas = _load(artifact, atlas_path, parcellation)
    if demo_model or demo_atlas:
        what = " and ".join(w for w, on in (("model", demo_model), ("atlas", demo_atlas)) if on)
        st.error(DEMO_BANNER.format(what=what), icon="🧪")
    st.warning(LIMITATIONS, icon="⚠️")

    choice = st.selectbox("Example molecule", ["(custom)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    smiles = st.text_input("SMILES", value=default or EXAMPLES["Nicotine"]).strip()
    if not smiles:
        st.stop()

    try:
        direction = parse_direction_spec(direction_spec) if direction_spec.strip() else None
        pred = predict_molecule(smiles, bundle, atlas, direction=direction,
                                weight_mode=weight_mode)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if pred.out_of_domain:
        st.warning(f"Out of applicability domain: nearest training molecule has Tanimoto "
                   f"{pred.max_training_tanimoto:.2f}. Treat predictions as unreliable.")
    elif np.isfinite(pred.max_training_tanimoto):
        st.caption(f"In domain (nearest training Tanimoto {pred.max_training_tanimoto:.2f}).")

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Predicted affinity (pChEMBL)")
        table = pd.DataFrame({
            "receptor": pred.keys,
            "pChEMBL": [pred.affinity[k] for k in pred.keys],
            "uncertainty (±)": pred.uncertainty,
        })
        st.bar_chart(table.set_index("receptor")["pChEMBL"])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(pred.projection.note)
    with right:
        st.subheader("Predicted brain map")
        scores = pred.region_scores
        parc = rc.parcellation
        html = None
        if not parc.subcortex and parc.name.startswith("Schaefer") and not demo_atlas:
            html = _surface_html(tuple(float(v) for v in scores))
        elif demo_atlas:
            st.info("Surface rendering is disabled in DEMO MODE: synthetic atlas columns do "
                    "not correspond to real cortical parcels.")
        if html:
            st.components.v1.html(html, height=460, scrolling=True)
        else:
            if "surface_error" in st.session_state:
                st.info(f"Surface unavailable ({st.session_state['surface_error']}); "
                        "showing region scores instead.")
            st.bar_chart(pd.DataFrame({"score": scores}, index=pd.RangeIndex(len(scores))))
        order = np.argsort(scores)[::-1]
        st.write("Top regions:", ", ".join(f"{int(r)} ({scores[r]:+.2f})" for r in order[:8]))


main()
