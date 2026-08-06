# Kaggle-based dataset build (webcam-realistic replacement for WIDER FACE)

Why: the first pass of this eval (`../frames`, `../labels.csv`) used WIDER FACE,
which is general event photography, not webcam footage of someone seated at a
laptop — the actual proctoring use case. This replaces/supplements it with the
**MSU Online Exam Proctoring (OEP) dataset**
([Kaggle](https://www.kaggle.com/datasets/raajanwankhade/oep-dataset),
Atoum et al., IEEE TMM 2017): real students and actors taking an exam on
camera, webcam mounted above the monitor, ~11.8GB of video across 24 subjects.
Too large to download wholesale and process locally, so this runs as a Kaggle
notebook against the dataset mounted directly in Kaggle's environment —
nothing large gets downloaded to a local machine.

The Kaggle account used here (`tanvishdesai`) has Dataset API access but not
Kernel API access (Kaggle gates the Kernels API behind phone verification),
so these notebooks are **pushed and run manually** through the Kaggle web UI
rather than via `kaggle kernels push`.

## Status: done. Results are in `../REPORT.md`

Ground truth (how many faces are *actually* in a frame) can't come from the
detector we're testing — scoring a model against labels it produced itself is
circular. It also can't come from the OEP `gt.txt` files: those label
*cheating-behavior types* (gaze, text, phone, etc. — confirmed by decoding a
sample video locally), not face count. So this ran as two steps:

1. **`phase1_extract_candidates.ipynb`** (run manually on Kaggle, output in
   `results/`): mounted the OEP dataset directly in Kaggle's environment,
   decoded 13 subjects' webcam videos, sampled a frame every 2.5s, and ran a
   quick Python MediaPipe pass **only** to flag candidate rare frames
   (0-face / 2+-face) worth a human look — `detector_predicted_count` in
   `results/candidate_manifest.csv` was never used as a label, only to pick
   which of ~1955 candidates were worth reviewing. Output: 79 contact-sheet
   grids (`results/contact_sheets/`).
2. **Human review** (`build-oep-labels.mjs`): every 0-face and 2+-face
   candidate sheet was reviewed by hand, plus a spread of the 1-face pool,
   and each frame's actual face count was recorded from what's visible in
   the image (a person's back/shoulder at the frame edge doesn't count as a
   "face" — only clearly visible facial features do). Selection rationale
   for every included frame is a code comment in that script.

Since the reviewed frames are small (154 JPEGs, not the 11.8GB source), the
eval itself runs locally via `../run-eval.mjs` against the merged
`../labels.csv` — no need for a separate Kaggle eval notebook once the
frames themselves are local. The selected frames now live in `../frames/`
(prefixed `oep_`); `results/` here only keeps the small provenance CSVs
(`candidate_manifest.csv` — every candidate the notebook flagged and what it
predicted; `sheet_manifest.csv` — which contact sheet/position each candidate
appeared at) since the raw candidate frame dump and contact-sheet images
(94MB, ~1955 mostly-unused images) aren't worth carrying in the repo once
review is done.

To regenerate from scratch: re-run phase 1 on Kaggle (steps below) into a
fresh `results/`, then `node build-oep-labels.mjs` (copies the selected
frames into `../frames/` and rebuilds `../labels.csv`'s OEP rows) and
`node ../run-eval.mjs`.

### Re-running phase 1

1. On kaggle.com: **New Notebook** → **File → Import Notebook** → upload
   `phase1_extract_candidates.ipynb`.
2. **Add Data** (right sidebar) → search `MSU Online Exam Proctoring Dataset`
   → add `raajanwankhade/oep-dataset`.
3. Settings → internet **on** (needed for `pip install mediapipe` if it isn't
   already in the base image).
4. **Run All**. Takes roughly 20–40 minutes (13 subjects, sampled every 2.5s,
   capped at 15 min/video) — CPU only, no GPU needed.
5. Download the **Output** tab's contents into `results/` here.
