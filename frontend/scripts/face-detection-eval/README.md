# Face-Count Detection Accuracy Eval

Offline accuracy evaluation for the proctoring face-count detector
(`frontend/src/components/ai/FaceDetectorWorker.ts`). See
[REPORT.md](./REPORT.md) for results and the acceptance decision.

Not part of the app build — a standalone tool with its own `package.json`,
run manually.

The dataset (`frames/`, `labels.csv`, 211 frames) is real exam-webcam footage
plus two small non-webcam edge-case sources OEP structurally can't supply:
171 frames hand-labeled from the MSU OEP dataset (via the Kaggle pipeline in
[kaggle/](./kaggle/README.md)) — the actual use case — 20 external
mask-condition frames (OEP predates mask-wearing as routine exam attire), and
20 real OEP frames with a synthetic exposure transform applied for the
`lighting` condition (OEP's webcam auto-exposure means it has no genuine
poor-lighting frames either — see REPORT.md's "Methodology" and
"Limitations" for the full reasoning, including why a real external
low-light dataset was evaluated and rejected). An earlier pass of this eval
also included WIDER FACE (general event photography); it was dropped
entirely as not representative of the actual use case.

## Layout

```
labels.csv                    frame_id, ground_truth_face_count, condition_tags, ...
frames/                       the 211 labeled test frames (171 OEP webcam + 20 mask + 20 synthetic lighting)
run-eval.mjs                  runs the production model config against labels.csv
results.json                  eval output (metrics + per-frame + per-condition predictions)
REPORT.md                     write-up: methodology, results, decision

build-dataset-mask.mjs        regenerates the mask-condition frames (fetches from HF directly, no download needed)
mask-selection.json           provenance: exact source image per mask frame
lib/                          shared parsing/CSV helpers

kaggle/                       OEP dataset: extraction notebook + human-reviewed labels, see kaggle/README.md
                               (also kaggle/build-synthetic-lighting.mjs, the lighting-condition generator)
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

Only needed if you want to change the frame selection — not needed to just
re-run the eval.

- **Mask frames**: `node build-dataset-mask.mjs` — fetches directly from a
  public HuggingFace dataset, no download needed. Writes `mask-selection.json`
  and the `mask_*.jpg` frames, but does not merge into `labels.csv` itself
  (append manually, or follow the pattern in `kaggle/build-oep-labels.mjs`).
- **OEP frames**: see [kaggle/README.md](./kaggle/README.md) — requires
  re-running the Kaggle notebook (phone-verification-gated Kernels API, so
  it's a manual web-UI run, not `kaggle kernels push`) or, if the raw
  candidate frame dump (`kaggle/results/frames/`) is already available
  locally, re-running `node kaggle/build-oep-labels.mjs` directly against it.
- **Synthetic lighting frames**: `node kaggle/build-synthetic-lighting.mjs` —
  also needs `kaggle/results/frames/` locally (same raw candidate dump as
  above). Picks previously-unused single-face candidates and writes
  underexposed/overexposed versions straight into `../frames/`, merging into
  `labels.csv` the same dedup-safe way `build-oep-labels.mjs` does.

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
