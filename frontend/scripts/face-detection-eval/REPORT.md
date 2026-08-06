# Face-Count Detection Accuracy — Test Report

Issue: [#1222 — Test & measure accuracy of face count detection](https://github.com/vicharanashala/vibe/issues/1222)

## Decision

**Current accuracy is not acceptable for proctoring as configured today —**
and it's worse than it first looked. An initial pass of this eval used WIDER
FACE (a general face-detection benchmark: event photography, not webcam
footage) and found a concerning but survivable `NO_FACE` precision of 0.342.
Adding a second, webcam-realistic test set — 154 frames hand-labeled from the
MSU Online Exam Proctoring dataset (real students, webcam mounted above the
monitor, an actual exam session) — drops `NO_FACE` precision to **0.060** on
that set alone. In plain terms: of every anomaly this model would have logged
as "student left the frame" against real exam-webcam footage in this test
set, **94% were wrong** — the student had a face in frame the whole time; the
model just didn't find it, usually because they were looking down, wearing a
cap, or resting a hand near their face — completely ordinary things a person
does while taking an exam.

| Source | n | `NO_FACE` precision | `NO_FACE` recall | `MULTIPLE_FACES` precision | `MULTIPLE_FACES` recall |
| --- | --- | --- | --- | --- | --- |
| WIDER FACE (event photography) | 272 | 0.342 | 0.975 | 0.821 | 0.582 |
| **OEP (real exam webcam)** | **154** | **0.060** | 1.000 | 1.000 | 0.556 |
| Combined | 426 | 0.238 | 0.977 | 0.857 | 0.575 |

Recommendation unchanged in kind, stronger in urgency: do not ship `NO_FACE`
as a single-frame hard pause/rewind trigger. See [Recommendations](#recommendations).

## Methodology

### Model under test

The exact configuration used in production
(`frontend/src/components/ai/FaceDetectorWorker.ts`):

```
@tensorflow-models/face-detection, SupportedModels.MediaPipeFaceDetector
runtime: "tfjs", modelType: "full", maxFaces: 10
```

Run via `run-eval.mjs` on the `@tensorflow/tfjs-backend-wasm` CPU backend
(the Node equivalent of the browser CPU-fallback path; `@tensorflow/tfjs-node`
could not be used in this environment — see [Limitations](#limitations)).
Input frames are decoded with `jpeg-js` and capped to 1280px on the long edge
before inference, matching a realistic webcam frame scale.

### Anomaly bucketing

Reproduces `floating-video.tsx`'s existing mapping so the metrics mean the
same thing the production anomaly flags mean:

| Detected face count | Anomaly            |
| -------------------- | ------------------- |
| 0                     | `NO_FACE`           |
| 1                     | none (OK)            |
| 2+                    | `MULTIPLE_FACES`    |

### Test set — 426 labeled frames (`labels.csv`)

Two sources, built two different ways:

**WIDER FACE (272 frames)** — real photos with **ground-truth face counts and
per-face attributes** (blur, illumination, occlusion, pose) from the
dataset's own annotations, used directly as condition tags; no manual
labeling. General event photography, not webcam footage — this is why a
second source was added.

**MSU Online Exam Proctoring / "OEP" dataset (154 frames)** — real students
and actors actually taking an exam on camera, webcam mounted above the
monitor: the actual use case. At ~11.8GB across 24 subjects it's too large to
download and process locally, so it was handled as a two-phase Kaggle
pipeline (`kaggle/`):

1. **Phase 1** (`kaggle/phase1_extract_candidates.ipynb`, run manually on
   Kaggle — the account used has Dataset API access but the Kernels API
   requires phone verification Kaggle hasn't granted it): mounts the dataset
   directly in Kaggle's environment (nothing large downloaded locally),
   decodes 13 subjects' webcam videos sequentially (prioritizing the 9 "real
   exam" subjects, where the proctor actually walks up to and talks to the
   student — the best real source of genuine multi-person frames), samples a
   frame every 2.5s, and runs a quick Python MediaPipe pass **only** to flag
   candidate rare frames (0-face / 2+-face) worth a human look. Output:
   contact-sheet grids of labeled thumbnails (79 sheets, ~1955 candidates).
2. **Human review** (`kaggle/build-oep-labels.mjs` encodes the result): every
   0-face and 2+-face candidate sheet was reviewed by hand, plus a spread of
   the 1-face pool, and each frame's *actual* face count was recorded from
   what's visible in the image — not from what either detector guessed
   (scoring a model against labels it produced itself would be circular, and
   the OEP dataset's own `gt.txt` files turned out to label cheating-*behavior*
   types — gaze, text, phone use — not face count, confirmed by decoding a
   sample video). A frame only counts a face as present if facial features
   (eyes/nose/mouth) are actually visible — a person's back or shoulder at
   the frame edge doesn't count as a "face."

Composition (both sources combined):

| condition_tags | n | ground truth | source |
| --- | --- | --- | --- |
| normal | 105 | 1 face, clean/typical | both |
| glasses | 50 | 1 face, wearing glasses | OEP |
| peer_present | 73 | 2–4 faces | both |
| angle | 36 | 1 face, atypical pose | both |
| occlusion | 32 | 1 face, heavy occlusion | both |
| occlusion;angle | 19 | 1 face, cap pulled low **and** chin tucked down | OEP |
| background_crop | 40 | 0 faces (cropped from WIDER FACE, verified against its boxes) | WIDER |
| mask | 20 | 1 face, wearing a mask | WIDER (external mask dataset) |
| lighting | 20 | 1 face, extreme illumination | WIDER |
| partial_occlusion | 15 | 1 face, partial occlusion | WIDER |
| blur | 12 | 1 face, heavy blur | WIDER |
| absent | 4 | 0 faces, person stepped away from desk | OEP |

Full provenance: `wider-selection.json`, `mask-selection.json`,
`negatives-selection.json` (WIDER-derived) and `kaggle/results/candidate_manifest.csv`
+ `kaggle/build-oep-labels.mjs` (OEP — every included frame's selection
rationale is a code comment there). `labels.csv` is the canonical
`frame_id, ground_truth_face_count, condition_tags` file the eval reads.

Neither raw source (WIDER FACE's ~366MB archives, OEP's ~11.8GB video) is
committed — only the 426 selected/derived frames (`frames/`, ~30MB total).
See `README.md` and `kaggle/README.md` to regenerate either.

## Results

### Overall (n=426)

| Metric | Value |
| --- | --- |
| Bucket accuracy (NO_FACE / OK / MULTIPLE_FACES) | 61.5% |
| `NO_FACE` precision / recall / F1 | 0.238 / 0.977 / 0.382 |
| `MULTIPLE_FACES` precision / recall / F1 | 0.857 / 0.575 / 0.689 |

Confusion matrix (rows = ground truth, columns = predicted):

| | → NO_FACE | → OK | → MULTIPLE_FACES |
| --- | --- | --- | --- |
| **NO_FACE** (n=44) | 43 | 1 | 0 |
| **OK** (n=309) | 125 | 177 | 7 |
| **MULTIPLE_FACES** (n=73) | 13 | 18 | 42 |

### By source — this is the headline result

| Source | n | Bucket accuracy | `NO_FACE` P / R / F1 | `MULTIPLE_FACES` P / R / F1 |
| --- | --- | --- | --- | --- |
| WIDER FACE (event photography) | 272 | 64.3% | 0.342 / 0.975 / 0.506 | 0.821 / 0.582 / 0.681 |
| **OEP (real exam webcam)** | **154** | **56.5%** | **0.060 / 1.00 / 0.113** | 1.00 / 0.556 / 0.714 |

The general-photography benchmark *understated* how unreliable `NO_FACE` is
in the actual use case. On real exam-webcam frames, `NO_FACE` precision is
essentially a coin flip gone wrong — worse than random with respect to how
often the flag is *right*, because it's dominated by ordinary in-exam
behavior (looking down at the keyboard/paper, a hand near the chin, a cap)
rather than the student actually leaving.

### By condition (combined)

| Condition | n | Bucket accuracy | False `NO_FACE` rate | False `MULTIPLE_FACES` rate |
| --- | --- | --- | --- | --- |
| absent (true 0-face, OEP) | 4 | 100.0% | 0.0% | 0.0% |
| background_crop (true 0-face, WIDER) | 40 | 97.5% | 0.0% | 0.0% |
| normal | 105 | 85.7% | 11.4% | 2.9% |
| lighting | 20 | 75.0% | 15.0% | 10.0% |
| glasses | 50 | 70.0% | 30.0% | 0.0% |
| peer_present (true 2+) | 73 | 57.5% | 17.8%¹ | — |
| angle | 36 | 44.4% | 55.6% | 0.0% |
| occlusion | 32 | 34.4% | 65.6% | 0.0% |
| partial_occlusion | 15 | 33.3% | 60.0% | 6.7% |
| blur | 12 | 25.0% | 66.7% | 8.3% |
| mask | 20 | 10.0% | 90.0% | 0.0% |
| **occlusion;angle (cap + chin-down, OEP)** | **19** | **0.0%** | **100.0%** | 0.0% |

¹ For `peer_present`, "false `NO_FACE` rate" counts 2+-face frames the model
collapsed all the way down to 0 detections, not just undercounted.

Both true-0-face conditions (`absent`, `background_crop`) score near-perfect
— the model is not trigger-happy about seeing faces that aren't there. Every
failure mode is a **miss**, not a hallucination. The worst real-webcam
condition, `occlusion;angle` (one subject's cap-pulled-low-plus-chin-down
posture, sustained for nearly their entire ~15-minute session), missed
**every single frame**.

Raw per-frame predictions, the full misclassification list, and a per-source
breakdown are in `results.json`.

## Root cause

`NO_FACE` failures cluster exactly where a single-shot, no-tracking detector
would be expected to struggle: anything that hides the lower face (masks,
occlusion, a downward head tilt that puts the chin near the chest) or
degrades edge/contrast cues (blur, extreme lighting) pushes the detector's
internal confidence below its acceptance threshold on that one frame, and it
returns *zero* detections rather than a low-confidence one — no partial
credit. The OEP data shows this is not an edge case: looking down at a
keyboard or exam paper, or resting a hand near the chin, are things real
students do constantly, not rare conditions.

`MULTIPLE_FACES` recall (0.575 combined) fails the same way in reverse: a
second, harder-to-see face (partially out of frame, small, poor angle)
doesn't get detected, so the frame reads as 1 face instead of 2. Precision is
strong (0.857) — when it *does* fire, it's usually right.

**A cross-implementation finding surfaced during OEP labeling, worth flagging
explicitly:** the Python `mediapipe.solutions.face_detection` pass used on
Kaggle to help find review candidates (same nominal model — BlazeFace
full-range — different language runtime than production) persistently
double-counted several individuals as 2 faces where there was only 1 (one
subject: 222 of ~360 sampled frames). The production TFJS/WASM pipeline this
report actually evaluates does **not** reproduce that failure on those same
frames — instead it misses ~30% of them as `NO_FACE` (see `glasses`
condition above). Two runtimes of "the same" model disagree, on the same
input, about *how* they fail. Practical implication: accuracy characteristics
measured against one runtime/language binding of a model should not be
assumed to transfer to a different binding of the "same" model — re-test
after any such swap.

This all matters more because production reports an anomaly from a **single
frame** (`floating-video.tsx`, `handleImageAnomaly`) with no multi-frame
confirmation — one bad frame is enough to fire `NO_FACE`.

## Confidence scores — investigated, not available

Issue item #2 asks to persist per-face confidence instead of discarding it.
Investigating the actual model output found the premise doesn't hold for the
model in use: `@tensorflow-models/face-detection`'s `MediaPipeFaceDetector`
returns a `Face` object with only `box` and `keypoints` —
**no confidence score, in either the `"tfjs"` or `"mediapipe"` runtime**
(verified against the package's own `.d.ts` and by running the detector
directly; see `node_modules/@tensorflow-models/face-detection/dist/types.d.ts`).
Nothing is being discarded — the library never surfaces it.

What shipped instead:
- `faceCount` is now sent with every image anomaly report and persisted on
  the anomaly record (`backend/src/modules/anomalies/classes/validators/AnomalyValidators.ts`,
  `.../classes/transformers/Anomaly.ts`) — previously not captured at all.
- `confidenceScores` is accepted by the same endpoint (`number[]`, JSON-array-
  encoded for multipart requests) and stored if present, so a future detector
  swap can start populating it with no further schema change. It is not
  populated by the current frontend, and the frontend/backend both document
  why in code comments rather than sending a fabricated placeholder value.

Options to actually get confidence scores, for a future issue if useful:
1. Switch the detector to the newer `@mediapipe/tasks-vision` `FaceDetector`
   API, whose `Detection.categories[].score` does expose one — a bigger
   change (new dependency, different loading model in a Worker) than the
   scope of this issue.
2. Accept that this detector family reports a binary "found the face" signal
   per frame and lean on the multi-frame confirmation in
   [Recommendations](#recommendations) instead of a per-frame confidence
   threshold.

## Recommendations

1. **Require N-of-M consecutive frames before firing `NO_FACE`.** This is no
   longer optional — on real exam-webcam footage, a single-frame trigger
   would false-flag a student for looking down at their exam paper. Requiring,
   e.g., 3 consecutive positive detections (already sampled every ~1s per
   `floating-video.tsx`'s throttle) before reporting would filter out
   single-frame misses while still catching a student who is genuinely gone.
2. **Do not pause/rewind on `MULTIPLE_FACES` from a single frame either** —
   0.575 recall means real peer-present cases are already under-caught; a
   single well-detected frame is reasonably trustworthy evidence (0.857
   precision) and is the right trigger for *logging*, but a stricter
   confirmation window makes sense before an automatic pause.
3. **Don't rely on model confidence for threshold tuning** — it isn't
   available from this model. Tune on detection *count stability* across
   consecutive frames instead (recommendation 1).
4. **If evaluating a different detector/runtime, re-run the full eval** —
   don't assume accuracy transfers across implementations of "the same"
   model (see the cross-implementation finding above). Use
   `run-eval.mjs` against `labels.csv` for a like-for-like comparison.

## Limitations

- `@tensorflow/tfjs-node` could not be built in this environment (no native
  toolchain / no prebuilt binary for the Node version available here — see
  `README.md`), so the eval runs on the WASM CPU backend rather than the
  native Node backend. Both are CPU inference paths using the same model
  weights and the same code path the browser falls back to when WebGL is
  unavailable (`FaceDetectorWorker.ts`); this does not change *what* the
  model does, only inference speed.
- WIDER FACE is general event photography, not webcam footage — that's why
  the OEP set was added. It remains useful as a cross-check with richer
  per-face attribute ground truth than could be hand-labeled at this scale.
- OEP frames are drawn from 13 of the dataset's 24 subjects, sampled every
  2.5s and capped at 15 minutes/video, so some rare conditions (e.g. genuine
  multi-person moments) are necessarily a small absolute count (n=73 across
  both sources) even though real. The Kaggle account's phone-unverified
  status meant the extraction notebook had to be run manually rather than
  pushed via API — see `kaggle/README.md` for exactly what was and wasn't
  reviewed (79 contact sheets covering ~1955 auto-flagged candidates; every
  0-face and 2+-face candidate was reviewed, plus a spread of the 1-face
  pool).
- The `mask` condition uses tightly-cropped face images (a mask-classification
  dataset), not full webcam-style scenes with background — the closest
  available real, individually-fetchable, ground-truthed mask data.
- `virtual_camera` is out of scope for this frame-level eval: it's a
  device/stream-metadata check (`frontend/src/utils/proctoring/detectVirtualCamera.ts`,
  label/resolution/frame-rate sniffing), not a pixel-content condition — a
  frame relayed through a virtual camera is pixel-identical to its source, so
  it doesn't change face-detection behavior and there is nothing distinct to
  label in a static-image test set for it.
