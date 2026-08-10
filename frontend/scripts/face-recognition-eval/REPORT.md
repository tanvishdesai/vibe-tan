# Face Recognition (Identity Verification) Accuracy — Report

Issue: [#1224](https://github.com/vicharanashala/vibe/issues/1224)

## Decision

**The current 0.55 Euclidean-distance threshold is not adequate as a security
control, and neither is any other single threshold on this model/pipeline.**
Equal Error Rate (EER) is **24.4%** — 5-12x above the 2-5% target the issue
cites for exam proctoring. At the shipped threshold (0.55), False Reject Rate
(FRR) is already 29.5% overall and climbs to **42.4%** for verification
attempts more than two minutes after enrollment; False Accept Rate (FAR) is
7.7%, meaning roughly 1 in 13 impostor attempts on this footage would be
silently accepted as the enrolled student. The matcher's overall
discriminative power on this footage is weak (ROC AUC 0.565 — only modestly
better than a coin flip), so this isn't a threshold-tuning problem: moving to
the empirically-optimal EER threshold (0.65) still leaves both error rates
around 24-25%.

| Metric | At current threshold (0.55) | At EER threshold (0.65) |
|---|---|---|
| FAR (impostor wrongly accepted) | 7.7% | 25.0% |
| FRR (genuine student wrongly rejected) | 29.5% | 23.7% |
| EER | — | 24.4% |
| ROC AUC | 0.565 | |

## Methodology

**Model under test**: `@vladmandic/face-api` — TinyFaceDetector
(`inputSize: 512, scoreThreshold: 0.45`) → FaceLandmark68Net →
FaceRecognitionNet (128-d descriptor), exactly the configuration in
`frontend/src/components/ai/FaceRecognitionComponent.tsx`. Match rule:
`euclideanDistance(reference, live) < MATCH_THRESHOLD` (0.55, hardcoded).
This eval runs the same model, same detector config, and the same distance
function against a held-out pair set — see `run-eval.mjs`.

**Dataset**: real exam-webcam footage, not a generic face-verification
benchmark (LFW etc.) — reused from the MSU Online Exam Proctoring (OEP)
dataset (Atoum et al., IEEE TMM 2017) already pulled down locally for the
face-count eval (#1222); see `build-pairs.mjs` and
[../face-detection-eval/kaggle/README.md](../face-detection-eval/kaggle/README.md)
for the original extraction story. 13 of OEP's subjects have a reviewable
local frame pool (~90-260 frames each, one every 2.5s across a session).

For each subject: the earliest usable frame becomes the **enrollment**
photo (mirrors the real one-time profile capture at
`/users/me/face-reference`), later frames of the same subject become
**genuine** probes, and other subjects' frames become **impostor** probes.
"Usable" means the frame has *exactly one* face at the production score
threshold — see "Methodology gotcha" below for why `detectSingleFace` (pick
the best face) was rejected in favor of `detectAllFaces` + `length === 1`.

| | Count |
|---|---|
| Subjects (identities) | 13 |
| Unique frames | 169 |
| Genuine pairs | 156 |
| Impostor pairs | 312 |
| Total pairs | 468 |
| Pairs skipped (re-detection failed) | 0 |

Condition tags bucket genuine pairs by elapsed time since enrollment within
the same recording session (OEP is one continuous session per subject, so
this is seconds/minutes, not days — see Limitations):

| Condition | n (genuine pairs) | Time delta |
|---|---|---|
| `genuine-near` | 23 | ≤ 30s |
| `genuine-mid` | 67 | 31-120s |
| `genuine-far` | 66 | > 120s |

**Methodology gotcha caught during dataset construction**: OEP frames
sometimes have a bystander in addition to the subject. The first pass used
`detectSingleFace` (returns whichever face scores highest) and produced a
near-chance ROC AUC (0.499). Inspecting the closest-in-time "genuine" pairs
showed why: in `oep_subject11_00000s.jpg` and `_00002s.jpg` the subject's
face was the top-scoring detection (0.968, 0.938), but in `_00004s.jpg`
(2 seconds later) a bystander's face edged it out (0.894 vs 0.860) — so that
"genuine" pair was actually comparing two different people. Switching to
`detectAllFaces` + requiring exactly one face above threshold (`lib/detect.mjs`)
fixed this and moved AUC to 0.565. This also better matches what the
production component expects (one person alone at their desk), so frames
with a visible bystander are correctly excluded rather than silently
mismatched.

## Results

### Overall

See the Decision table above. Full threshold sweep (0.30-0.80) is in
`results.json` (`thresholdSweep`).

### By condition (genuine-pair FRR)

| Condition | n | FRR @ 0.55 (current) | FRR @ 0.65 (EER) |
|---|---|---|---|
| `genuine-near` (≤30s) | 23 | 4.3% | 4.3% |
| `genuine-mid` (31-120s) | 67 | 25.4% | 20.9% |
| `genuine-far` (>120s) | 66 | 42.4% | 33.3% |

FRR degrades sharply with elapsed time, even *within a single ~10-minute
session* — the embedding is not very stable to the ordinary pose/lighting
drift of someone sitting at a desk, well before any cross-day drift (aging,
different lighting setup, different camera) that real enrollment-vs-exam-day
usage would add on top.

### Hardest impostor pairs (closest distance, i.e. most likely false accepts)

| Pair | Distance | Note |
|---|---|---|
| subject19 enrollment vs subject18 probe | 0.429 | see caveat below |
| subject18 enrollment vs subject19 probe | 0.449 | see caveat below |
| subject15 enrollment vs subject16 probe | 0.463 | |
| subject10 enrollment vs subject9 probe | 0.465 | |
| subject19 enrollment vs subject14 probe | 0.473 | |

**Caveat**: the subject18/subject19 pair's two source images
(`qc-review/hard-negative-impostor/`) look like they could plausibly be the
same person on visual inspection. OEP's public documentation doesn't
guarantee subject-ID uniqueness is preserved across all recording sessions,
and this wasn't independently verifiable from what's available locally. This
doesn't affect the near/mid/far genuine-pair results above (unrelated
subjects), and if it *is* a labeling collision, the effect on the headline
FAR number is conservative — a true match mislabeled as an impostor pair
inflates measured FAR rather than hiding risk.

## Root cause

Two independent things are true at once: (1) TinyFaceDetector/FaceRecognitionNet
via `@vladmandic/face-api` is a lightweight, several-years-old model choice,
and (2) this footage is genuinely hard for it — low resolution, webcam-typical
motion blur, and (per #1222's face-count eval on the same footage) a
detector that already struggles just to find a face reliably. ROC AUC of
0.565 says the 128-d embedding barely separates same-person from
different-person distances on this material at all, which is a modeling
ceiling, not a threshold-tuning problem — no single cutoff fixes an EER of
24%.

## Recommendations

1. **Don't treat FACE_RECOGNITION as an autonomous enforcement signal at the
   current threshold** — do not silently lock a student out based on a
   single distance comparison. FRR of 29-42% means real students would be
   frequently and wrongly flagged, especially later in an exam.
2. **Moving the threshold alone will not reach the issue's 2-5% EER target**
   on this model — the EER-optimal point is still 24.4%. Reaching the target
   likely requires a stronger face-embedding model (the FaceRecognitionNet
   bundled with `@vladmandic/face-api` is dated) and/or higher-resolution
   capture, not just a threshold change.
3. **Multiple reference images per enrollee** (the issue's optional ask) is
   worth prioritizing given how much FRR degrades with elapsed time even
   within one session — `IUser.faceEmbedding` currently stores exactly one
   embedding per user (`backend/src/modules/users/services/UserService.ts`).
   A small reference set spanning a few minutes/poses would likely reduce
   FRR meaningfully; this eval doesn't test that directly and it's a
   reasonable next increment.
4. Until (2) lands, consider surfacing FACE_RECOGNITION anomalies as a
   review queue item for a human proctor rather than an automatic
   consequence, given both FAR and FRR are too high for unattended decisions.

## Limitations

- **Time deltas are within-session (seconds/minutes), not cross-day.** OEP
  is one continuous recording per subject; there is no genuine multi-day
  recapture data in this source (same gap #1222 flagged for other
  conditions). The `genuine-far` bucket (>120s) is the closest proxy
  available, and even it likely understates real-world drift.
- **No masked/bespectacled pairs of a *known* identity.** OEP has no
  mask-wearing subjects, and the anonymous mask images used in #1222's eval
  aren't paired to any of these 13 identities, so they can't produce a
  genuine-match pair here.
- **13 identities instead of the issue's suggested 3-5.** Chosen
  deliberately for breadth — a security-relevant metric like FAR should
  reflect "any other student," not a fixed handful — at the cost of shipping
  fewer pairs per person (~12 genuine probes/subject) than the issue's
  100-200/person suggestion.
- **Hard negatives are discovered, not curated** — the closest-distance
  impostor pairs are reported as found rather than hand-picked for visual
  similarity; see the subject18/19 labeling caveat above.
- **Single embedding per user is a product constraint**, not a gap in this
  eval — see Recommendation 3.
- **`matchDistance` is only logged on confirmed mismatches in production**
  (`FaceRecognitionComponent.tsx`'s `REQUIRED_MISMATCHES = 2` gate) — this
  eval's FAR/FRR/EER come from running the model directly against the pair
  set, independent of what gets logged in prod.
