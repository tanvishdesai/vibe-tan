# Face-Count Detection Accuracy — Test Report

Issue: [#1222 — Test & measure accuracy of face count detection](https://github.com/vicharanashala/vibe/issues/1222)

## Decision

**Current accuracy is not acceptable for proctoring as configured today.**
On 211 frames of real exam-webcam footage (171 human-labeled from the MSU
Online Exam Proctoring dataset, 20 external mask-condition frames, and 20
synthetically-relit OEP frames — see [Methodology](#methodology)), `NO_FACE`
precision is **0.044** — of every anomaly this model would log as "student
left the frame," **95.6% are wrong**. The student had a face in frame the
whole time; the model just didn't find it, usually because they were looking
down, resting a hand near their face, wearing glasses, or wearing a mask —
completely ordinary things a person does while taking an exam.

| Metric | Value |
| --- | --- |
| n | 211 |
| Bucket accuracy (NO_FACE / OK / MULTIPLE_FACES) | 56.9% |
| `NO_FACE` precision / recall / F1 | 0.044 / 1.000 / 0.084 |
| `MULTIPLE_FACES` precision / recall / F1 | 1.000 / 0.600 / 0.750 |

Recommendation unchanged in kind: do not ship `NO_FACE` as a single-frame
hard pause/rewind trigger. See [Recommendations](#recommendations).

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

### Test set — 211 labeled frames (`labels.csv`)

**Real exam-webcam footage only.** An earlier pass of this eval also included
WIDER FACE (general event photography — press conferences, sports, crowds)
as a supplementary source. It was dropped: WIDER FACE isn't representative of
the actual use case (a single person seated at a laptop, webcam mounted above
the monitor), and mixing it in diluted the headline number — the WIDER-only
`NO_FACE` precision (0.342) was substantially *less* alarming than the
webcam-realistic number, which would have understated the real risk. Every
frame in this test set now comes from one of two real, ground-truthed
sources:

**MSU Online Exam Proctoring / "OEP" dataset (171 frames)** — real students
and actors actually taking an exam on camera, webcam mounted above the
monitor: the actual use case. At ~11.8GB across 24 subjects it's too large to
download and process locally, so it was handled as a two-phase Kaggle
pipeline (`kaggle/`):

1. **Phase 1** (`kaggle/phase1_extract_candidates.ipynb`, run manually on
   Kaggle — the account used has Dataset API access but the Kernels API
   requires phone verification Kaggle hasn't granted it): mounts the dataset
   directly in Kaggle's environment, decodes 13 of 24 subjects' webcam videos
   sequentially, samples a frame every 2.5s, and runs a quick Python
   MediaPipe pass **only** to flag candidate rare frames (0-face / 2+-face)
   worth a human look. Output: 1955 candidate frames, batched into 79
   contact-sheet grids of labeled thumbnails.
2. **Human review** (`kaggle/build-oep-labels.mjs` encodes the result): every
   0-face and 2+-face candidate sheet was reviewed by hand, plus a spread of
   the 1-face pool, and each frame's *actual* face count was recorded from
   what's visible in the image — not from what either detector guessed
   (scoring a model against labels it produced itself would be circular).
   A frame only counts a face as present if facial features (eyes/nose/mouth)
   are actually visible.

All 13 sampled subjects are now represented — one (`subject1`) was initially
skipped in favor of the 9 "real exam" subjects (where a proctor actually
walks up and talks to the student, the richest source of genuine multi-person
frames), then reviewed in a second pass via its own contact sheets. It
contributed a clean single-face stretch, several hand-over-mouth/chin
occlusion moments, downward reading-angle moments, and one genuine
peer-present event (a second person, clearly visible, confirmed not a
double-count artifact).

**Mask (20 frames)** — a face-mask edge case has no real analogue anywhere in
the 2017 OEP corpus (it predates mask-wearing as routine exam attire), so
this is sourced separately: tightly-cropped single-face images from a public
HF-hosted mirror of the Kaggle "Face Mask 12K Images" dataset
(`build-dataset-mask.mjs`). Not WIDER FACE, not webcam footage — the closest
available real, individually-fetchable, ground-truthed mask data.

**Lighting (20 frames)** — same problem as mask: OEP's webcam corpus has
consistent auto-exposure and contains no genuine poor-lighting frames
(confirmed by a quantitative scan — Laplacian variance for blur, mean/contrast
luminance for lighting extremity — over the 844 previously-unreviewed OEP
candidate frames; the most extreme candidates by that scan were still
visually normal, sharp, evenly-lit shots on manual check). A real external
low-light face dataset (DARK FACE, CVPR 2019 UG2+ Challenge) was evaluated
and rejected: it's outdoor nighttime crowd/surveillance photography
(~8.4 faces/image on average) with no license listed, a worse representation
of "webcam under poor lighting" than a synthetically degraded real webcam
frame. So instead, `kaggle/build-synthetic-lighting.mjs` takes real,
previously-unused single-face OEP frames and applies a controlled exposure
transform (heavy underexposure or overexposure, `synthlight_dark_*` /
`synthlight_bright_*`) — the face is genuinely there, ground truth is still
1, only the lighting is synthetic. Labeled as such in `source_dataset` so
it's never confused with a naturally-occurring case; see
[Limitations](#limitations).

Composition:

| condition_tags | n | ground truth | source |
| --- | --- | --- | --- |
| glasses | 50 | 1 face, wearing glasses | OEP |
| normal | 43 | 1 face, clean/typical | OEP |
| peer_present | 20 | 2 faces | OEP |
| mask | 20 | 1 face, wearing a mask | external (HF mirror of Kaggle Face Mask 12K) |
| lighting | 20 | 1 face, extreme illumination | OEP frame, synthetic exposure adjustment |
| angle | 20 | 1 face, atypical pose | OEP |
| occlusion;angle | 19 | 1 face, cap pulled low **and** chin tucked down | OEP |
| occlusion | 15 | 1 face, heavy occlusion | OEP |
| absent | 4 | 0 faces, person stepped away from desk | OEP |

Full provenance: `mask-selection.json` and `kaggle/results/candidate_manifest.csv`
+ `kaggle/build-oep-labels.mjs` / `kaggle/build-synthetic-lighting.mjs` (every
included OEP frame's selection rationale is a code comment there).
`labels.csv` is the canonical `frame_id, ground_truth_face_count,
condition_tags` file the eval reads.

The OEP source video (~11.8GB) and the mask dataset's own archive are not
committed — only the 211 selected/derived frames (`frames/`) are. See
`README.md` and `kaggle/README.md` to regenerate either.

## Results

### Overall (n=211)

| Metric | Value |
| --- | --- |
| Bucket accuracy | 56.9% |
| `NO_FACE` precision / recall / F1 | 0.044 / 1.000 / 0.084 |
| `MULTIPLE_FACES` precision / recall / F1 | 1.000 / 0.600 / 0.750 |

Confusion matrix (rows = ground truth, columns = predicted):

| | → NO_FACE | → OK | → MULTIPLE_FACES |
| --- | --- | --- | --- |
| **NO_FACE** (n=4) | 4 | 0 | 0 |
| **OK** (n=187) | 83 | 104 | 0 |
| **MULTIPLE_FACES** (n=20) | 4 | 4 | 12 |

`NO_FACE` recall is perfect (all 4 genuine absences caught) and precision is
essentially a coin flip gone badly wrong: of 91 frames the model flagged
`NO_FACE`, 87 had a real face in frame. `MULTIPLE_FACES` is the mirror image
— when it fires it's always right (precision 1.000), but it misses 8 of 20
genuine multi-person frames (recall 0.600).

### By condition

| Condition | n | Bucket accuracy | False `NO_FACE` rate |
| --- | --- | --- | --- |
| absent (true 0-face) | 4 | 100.0% | 0.0% |
| **lighting (synthetic exposure)** | **20** | **90.0%** | **10.0%** |
| normal | 43 | 79.1% | 20.9% |
| glasses | 50 | 70.0% | 30.0% |
| peer_present (true 2+)¹ | 20 | 60.0% | 20.0% |
| occlusion | 15 | 46.7% | 53.3% |
| angle | 20 | 40.0% | 60.0% |
| mask | 20 | 10.0% | 90.0% |
| **occlusion;angle (cap + chin-down)** | **19** | **0.0%** | **100.0%** |

`lighting` stands out: the model is far more robust to exposure extremes
(90% accuracy, only 2/20 missed) than to anything that obscures the lower
face. Read this alongside [Limitations](#limitations) — it's a synthetic
stress test on real faces, not naturally-occurring poor lighting, so treat it
as evidence the detector's failure mode is specifically face-occlusion, not
general image-quality degradation, rather than a literal "poor lighting is
fine in production" claim.

¹ For `peer_present`, "false `NO_FACE` rate" counts 2+-face frames the model
collapsed all the way down to 0 detections, not just undercounted (the
remaining 8/20 undercounted to exactly 1, not 0).

Both true-0-face conditions score near-perfect (`absent` 100%) — the model is
not trigger-happy about seeing faces that aren't there. Every failure mode is
a **miss**, not a hallucination. The worst condition, `occlusion;angle`
(cap-pulled-low-plus-chin-down posture, sustained for nearly one subject's
entire ~15-minute session), missed **every single frame**.

Raw per-frame predictions and the full misclassification list are in
`results.json`.

## Root cause

`NO_FACE` failures cluster exactly where a single-shot, no-tracking detector
would be expected to struggle: anything that hides the lower face (masks,
occlusion, a downward head tilt that puts the chin near the chest) pushes the
detector's internal confidence below its acceptance threshold on that one
frame, and it returns *zero* detections rather than a low-confidence one — no
partial credit. Looking down at a keyboard or exam paper, resting a hand near
the chin, or wearing glasses are things real students do constantly, not rare
conditions — and `mask` (90% false `NO_FACE` rate) shows the same pattern
against an even more common face-covering.

`MULTIPLE_FACES` recall (0.600) fails the same way in reverse: a second,
harder-to-see face (partially out of frame, small, poor angle) doesn't get
detected, so the frame reads as 1 face instead of 2. Precision is perfect
(1.000) — when it *does* fire, it's always right.

**A cross-implementation finding, worth flagging explicitly:** the Python
`mediapipe.solutions.face_detection` pass used on Kaggle to help find review
candidates (same nominal model — BlazeFace full-range — different language
runtime than production) persistently double-counted several individuals as
2 faces where there was only 1 (one subject: 222 of ~360 sampled frames). The
production TFJS/WASM pipeline this report actually evaluates does **not**
reproduce that failure on those same frames — instead it misses ~30% of them
as `NO_FACE` (see `glasses` condition above). Two runtimes of "the same"
model disagree, on the same input, about *how* they fail. Practical
implication: accuracy characteristics measured against one runtime/language
binding of a model should not be assumed to transfer to a different binding
of the "same" model — re-test after any such swap.

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
   0.600 recall means real peer-present cases are already under-caught; a
   single well-detected frame is fully trustworthy evidence (1.000 precision)
   and is the right trigger for *logging*, but a stricter confirmation window
   makes sense before an automatic pause.
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
- **n=211 is just above this issue's ~200–300 frame target range.**
- **`lighting` is synthetic, not naturally-occurring poor lighting.** OEP's
  webcam corpus has consistent auto-exposure and contains no genuine
  poor-lighting frames — confirmed by a quantitative scan (Laplacian variance
  for blur, mean/contrast luminance for lighting extremity) over the 844
  previously-unreviewed OEP candidate frames, whose most extreme candidates
  were still visually normal on manual check. A real external low-light
  dataset (DARK FACE) was evaluated and rejected — outdoor nighttime
  crowd/surveillance photos, ~8.4 faces/image, no license listed — a worse
  representation of "webcam under poor lighting" than a synthetically
  degraded real webcam frame. So `lighting` frames here are real OEP faces
  with a controlled exposure transform applied
  (`kaggle/build-synthetic-lighting.mjs`) rather than naturally-occurring
  cases. The 90% accuracy on this condition should be read as "the detector
  tolerates exposure extremes on this specific transform," not as a general
  claim about production behavior under real poor lighting.
- **`blur` has no representation at all** — unlike `lighting`, this was never
  a required edge case in the issue (its explicit list is masks, partial
  occlusion, side angles, poor lighting, virtual camera), and the same scan
  that ruled out natural blur examples in OEP means it would need its own
  synthetic or external treatment if ever prioritized.
- OEP frames are drawn from 13 of the dataset's 24 subjects, sampled every
  2.5s and capped at 15 min/video, so some rare conditions (e.g. genuine
  multi-person moments) are necessarily a small absolute count (n=20) even
  though real. The Kaggle account's phone-unverified status meant the
  extraction notebook had to be run manually rather than pushed via API —
  see `kaggle/README.md` for exactly what was and wasn't reviewed. A large
  additional pool (~1780 further candidate frames across the same 13
  subjects, plus 11 entirely unsampled subjects) exists if the raw candidate
  dump is available locally — see `kaggle/README.md`.
- `absent` (true 0-face) sits at n=4 and is unlikely to grow much further from
  this pool: essentially all of the pool's 0-face-predicted candidates were
  already reviewed, and the great majority turned out to be occlusion/angle
  misses (a face present but undetected), not genuine departures — itself a
  real finding about how rarely students fully leave frame during a recorded
  exam.
- The `mask` condition uses tightly-cropped face images (a mask-classification
  dataset), not full webcam-style scenes with background — the closest
  available real, individually-fetchable, ground-truthed mask data.
- `virtual_camera` is out of scope for this frame-level eval: it's a
  device/stream-metadata check (`frontend/src/utils/proctoring/detectVirtualCamera.ts`,
  label/resolution/frame-rate sniffing), not a pixel-content condition — a
  frame relayed through a virtual camera is pixel-identical to its source, so
  it doesn't change face-detection behavior and there is nothing distinct to
  label in a static-image test set for it.
