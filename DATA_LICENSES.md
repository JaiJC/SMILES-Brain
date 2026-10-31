# Data sources and licences

The **code** in this repository is MIT licensed (`LICENSE`). The **data** it uses comes from
third parties under different terms. This file records what each source is, whether this
repository redistributes it, and what the terms are. It is a working record, **not legal
advice**; confirm anything you rely on commercially with the data owners.

| Source | Used for | Redistributed in this repo? | Terms | Verified |
|---|---|---|---|---|
| ChEMBL | affinity labels (`data/affinity_merged/`, `data/affinity_pdsp/` contain ChEMBL-derived rows) | **Yes** (derived per-receptor tables) | CC BY-SA 3.0: attribution and share-alike apply to the redistributed data | From ChEMBL's published licence; not re-checked in this session |
| PDSP Ki Database (UNC) | additional Ki labels (merged into the files above) | **Yes** (derived rows) | Not verified here. Check the PDSP site for terms and citation requirements before redistributing or commercial use | **No** |
| Hansen et al. 2022 receptor atlas (`netneurolab/hansen_receptors`) | receptor density maps (`data/atlas/`, gitignored) | No (downloaded at run time) | **CC BY-NC-SA 4.0: non-commercial.** Commercial use of the atlas, or of maps derived from it, needs the authors' permission | Read from the upstream `LICENSE.md` |
| Schaefer 2018 parcellation (CBIG) | parcel labels, centroids | No (downloaded at run time) | MIT (Copyright 2016 CBIG) | Read from the upstream `LICENSE.md` |
| Dukart et al. 2018, *Sci Rep* supplement | seven drug CBF maps (`data/validation/`, gitignored) | No; only derived summary tables are in `docs/results/` | Check the article's licence page (Scientific Reports articles are normally open access under CC BY) | **No** |
| Carhart-Harris et al. 2016, *PNAS*, NeuroVault collection 1083 | LSD CBF map (gitignored) | No; only derived summary tables | NeuroVault shows a licence on the collection page; not exposed in the API; check it | **No** |

## What this means in practice

- **Non-commercial atlas.** Anything that depends on the Hansen atlas (the projection stage, the
  brain-map view, all validation against drug maps) inherits its non-commercial restriction
  unless the authors permit otherwise. The receptor-profile predictor (labels -> model) does not
  use the atlas, but is trained on ChEMBL/PDSP-derived data (see the next point).
- **Share-alike data.** If you redistribute `data/affinity_merged/` or `data/affinity_pdsp/`,
  keep the ChEMBL attribution and release them under CC BY-SA 3.0 (or a compatible licence).
  The MIT licence on the code does not override that for those folders.
- **Trained artifacts** (`artifacts/*.joblib`) are derived from the labels above and are
  gitignored. Whether a model trained on CC BY-SA data must itself be share-alike is a legal
  question this file does not answer.
- **Cite the sources** in any write-up: ChEMBL (Zdrazil et al.), PDSP (Roth lab, UNC), Hansen et
  al. 2022 (*Nature Neuroscience*), Schaefer et al. 2018 (*Cerebral Cortex*), Dukart et al.
  2018 (*Scientific Reports*), Carhart-Harris et al. 2016 (*PNAS*).
