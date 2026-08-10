# Face Recognition (Identity Verification) Accuracy Eval

Offline accuracy evaluation for the proctoring identity matcher
(`frontend/src/components/ai/FaceRecognitionComponent.tsx`). See
[REPORT.md](./REPORT.md) for results and the acceptance decision.

Not part of the app build — a standalone tool with its own `package.json`,
run manually.

The dataset (`frames/`, `pairs.csv`, 169 frames / 468 pairs) is real
exam-webcam footage from the MSU Online Exam Proctoring (OEP) dataset,
reused from the frame pool already downloaded locally for the face-count
eval (#1222) — see `../face-detection-eval/kaggle/README.md` for the
original extraction story. No new dataset acquisition was needed: identity
ground truth comes directly from OEP's filename-encoded subject IDs
(`oep_subjectN_TTTTTs.jpg`), not manual review.

## Layout

```
pairs.csv        pair_id, person_id, ground_truth_match, condition_tags, time_delta_s, ...
frames/           the 169 frames referenced by pairs.csv (13 subjects)
lib/detect.mjs    shared face-api.js Node setup (model loading, tensor decode, single-face descriptor)
lib/csv.mjs       shared CSV read/write helpers
build-pairs.mjs   regenerates pairs.csv + frames/ from the local OEP frame pool
run-eval.mjs      runs the production model config against pairs.csv, computes FAR/FRR/EER/ROC
results.json      eval output (metrics, threshold sweep, per-condition, hard negatives)
qc-review/        sampled genuine/impostor pairs by bucket, for manual sanity-checking
REPORT.md         write-up: methodology, results, decision
```

## Re-run the eval

```
cd frontend/scripts/face-recognition-eval
npm install
npm run eval        # or: node run-eval.mjs
```

`pairs.csv` and `frames/` are committed — you don't need to regenerate the
dataset to re-run the eval. Takes under a minute on CPU (WASM backend).

## Regenerate the dataset from scratch

Only needed if you want to change frame/pair selection — requires the raw
OEP candidate frame dump (`../face-detection-eval/kaggle/results/frames/`,
~1955 frames) to already be present locally (kept local-only, not committed
— same convention as #1222's eval). If present:

```
node build-pairs.mjs
```

This picks the earliest usable frame per subject as the "enrollment" photo,
pairs it against later same-subject frames (genuine) and other subjects'
frames (impostor), and copies the selected frames into `frames/`. "Usable"
requires *exactly one* detected face at the production score threshold —
see the "Methodology gotcha" note in REPORT.md for why (an earlier version
used `detectSingleFace`, which silently mismatched people when a bystander
was also in frame).

## Why WASM instead of `@tensorflow/tfjs-node`, and why `@vladmandic/face-api`'s Node build

Same reasoning as `../face-detection-eval`: `@tensorflow/tfjs-node`'s native
addon isn't available to compile in this environment, so this uses the WASM
CPU backend instead. `@vladmandic/face-api` ships a dedicated
`dist/face-api.node-wasm.js` build for exactly this (Node + WASM, no
canvas/DOM) — `lib/detect.mjs` imports that entry point directly and runs
detection/recognition against raw `tf.tensor3d` input decoded with
`jpeg-js`, the same pattern `../face-detection-eval/run-eval.mjs` used for
the other model.
