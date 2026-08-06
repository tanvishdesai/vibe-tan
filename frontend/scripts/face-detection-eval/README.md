# Face-Count Detection Accuracy Eval

Offline accuracy evaluation for the proctoring face-count detector
(`frontend/src/components/ai/FaceDetectorWorker.ts`). See
[REPORT.md](./REPORT.md) for results and the acceptance decision.

Not part of the app build — a standalone tool with its own `package.json`,
run manually.

The dataset (`frames/`, `labels.csv`, 426 frames) combines two sources: WIDER
FACE (real annotated data, but general event photography) and 154 frames
hand-labeled from real exam-webcam footage (the MSU OEP dataset, via the
Kaggle pipeline in [kaggle/](./kaggle/README.md)) — the actual use case. See
REPORT.md's "By source" table: the webcam-realistic subset alone shows
`NO_FACE` precision of 0.06, notably worse than WIDER FACE suggested on its
own.

## Layout

```
labels.csv                    frame_id, ground_truth_face_count, condition_tags, ...
frames/                       the 426 labeled test frames (272 WIDER-derived + 154 OEP webcam)
run-eval.mjs                  runs the production model config against labels.csv
results.json                  eval output (metrics + per-frame + per-source predictions)
REPORT.md                     write-up: methodology, results, decision

build-dataset-wider.mjs       regenerates the WIDER FACE-derived frames (needs raw WIDER FACE download)
build-dataset-negatives.mjs   regenerates the 0-face crops (needs raw WIDER FACE download)
build-dataset-mask.mjs        regenerates the mask-condition frames (fetches from HF directly, no download needed)
build-labels-csv.mjs          merges the three *-selection.json manifests into labels.csv
lib/                          shared parsing/CSV helpers
*-selection.json              provenance: exact source image (+ crop box, for negatives) per frame

kaggle/                       webcam-realistic dataset: extraction notebook + human-reviewed labels, see kaggle/README.md
```

`frames/` and `labels.csv` are committed — you don't need to regenerate
anything to re-run the eval.

## Re-run the eval

```
cd frontend/scripts/face-detection-eval
npm install
npm run eval        # or: node run-eval.mjs
```

Takes under a minute on CPU (WASM backend). Writes `results.json` and prints
a summary table.

## Regenerate the dataset from scratch

Only needed if you want to change the frame selection (different seed,
different bucket sizes, etc.) — not needed to just re-run the eval.

1. Download the WIDER FACE validation split and annotations:
   ```
   curl -L -o WIDER_val.zip https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip
   curl -L -o wider_face_split.zip https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip
   unzip WIDER_val.zip && unzip wider_face_split.zip
   ```
   License: CC BY-NC-ND 4.0 (non-commercial). ~366MB combined — not committed
   to this repo.
2. Run the three builders, then merge:
   ```
   node build-dataset-wider.mjs      <path>/WIDER_val/images <path>/wider_face_split/wider_face_val_bbx_gt.txt
   node build-dataset-negatives.mjs  <path>/WIDER_val/images <path>/wider_face_split/wider_face_val_bbx_gt.txt
   node build-dataset-mask.mjs
   node build-labels-csv.mjs
   ```
   `build-dataset-wider.mjs` uses a fixed PRNG seed, so re-running against the
   same WIDER FACE download reproduces the same frame selection.

## Why WASM instead of `@tensorflow/tfjs-node`

`@tensorflow-models/face-detection` is also usable from Node via
`@tensorflow/tfjs-node` (already a `frontend` dependency, just unused
elsewhere in the app). It needs a native addon (`tfjs_binding.node`) that
either ships as a prebuilt binary for your Node version or gets compiled with
node-gyp. Neither was available in the environment this eval was built in
(no prebuilt binary for the installed Node version, no MSVC/Python build
toolchain), so the eval uses `@tensorflow/tfjs-backend-wasm` instead — pure
WebAssembly, no native compilation, same model weights and the same
CPU-inference code path the browser uses when WebGL is unavailable. If you
have a working `tfjs-node` build in your environment, swapping the backend in
`run-eval.mjs` (`tf.setBackend('wasm')` → `require('@tensorflow/tfjs-node')`)
will run faster but shouldn't change the results.
