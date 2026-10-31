# Design-screen demo (pipeline check, not a result)

`scripts/design_screen.py` on the real trained ridge artifact and the real atlas, run twice with
`--seed 0`; the two output CSVs are byte-identical (the P5-T2 reproducibility check).

- Library: `docs/results/design_library.txt` (8 catalog drugs + nicotine).
- Target profile: +1 on the 10 regions with the highest D2 density (24,25,31,32,37,41,75,79,90,91), -1 on the 10 lowest
  (4,5,14,53,54,63,65,85,86,98). This target was picked only to exercise the code.
- Output: `docs/results/design_screen_demo.csv`, ranked by pattern-match score in [-1, 1].

**How to read it.** 6 of 9 molecules have Tanimoto 1.0 to a training molecule
(they are in ChEMBL), so their predicted profiles reflect their own training labels. That makes
this a demonstration that the screen runs and is reproducible, not a screen of novel chemistry.
The score measures how well the predicted *receptor pattern* matches weights solved from the
target regions. Because affinity has not been shown to predict regional brain effects
(`docs/REPORT.md` 3.3-3.5), a high score is **not** evidence that a molecule produces that
regional effect. Everything here is an unvalidated hypothesis.
