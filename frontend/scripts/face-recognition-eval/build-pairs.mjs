// Builds the verification-pairs test set for issue #1224 from the MSU Online
// Exam Proctoring (OEP) frame pool already pulled down locally for the
// face-count eval (#1222) -- see ../face-detection-eval/kaggle/README.md for
// the extraction story. That pool is real exam-webcam footage across 13
// subjects with a filename-encoded subject + timestamp
// (oep_subjectN_TTTTTs.jpg), so identity ground truth needs no manual
// review here, unlike the face-count eval's face-count review.
//
// For each subject: picks the earliest usable frame as the "enrollment"
// photo (mirrors the real one-time profile-photo capture in
// UserController.ts's /users/me/face-reference), then pairs it against
// later same-subject frames (genuine) and other subjects' frames
// (impostor). "Usable" means face-api's TinyFaceDetector (production
// config) finds exactly one clear face -- this is a data-quality filter,
// not a ground-truth step, since identity is filename-derived either way.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeCsv, readCsv } from "./lib/csv.mjs";
import { loadModels, describeFace } from "./lib/detect.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OEP_FRAMES_DIR = path.join(__dirname, "../face-detection-eval/kaggle/results/frames");
const OEP_MANIFEST = path.join(__dirname, "../face-detection-eval/kaggle/results/candidate_manifest.csv");
const OUT_FRAMES_DIR = path.join(__dirname, "frames");
const OUT_PAIRS_CSV = path.join(__dirname, "pairs.csv");

const USABLE_PER_SUBJECT = 13; // 1 enrollment + up to 12 probes
const MAX_SCAN_PER_SUBJECT = 200; // give up on a subject after scanning this many candidates
const IMPOSTOR_PROBES_PER_OTHER_SUBJECT = 2;
const SOURCE_DATASET = "MSU_OEP (Atoum et al., IEEE TMM 2017) via Kaggle";
const LICENSE = "see-source-dataset";

function timeDeltaBucket(deltaS) {
  if (deltaS <= 30) return "genuine-near";
  if (deltaS <= 120) return "genuine-mid";
  return "genuine-far";
}

function groupBySubject(manifestRows) {
  const bySubject = new Map();
  for (const row of manifestRows) {
    const list = bySubject.get(row.subject) ?? [];
    list.push({ frame_id: row.frame_id, timestamp_s: Number(row.timestamp_s) });
    bySubject.set(row.subject, list);
  }
  for (const list of bySubject.values()) list.sort((a, b) => a.timestamp_s - b.timestamp_s);
  return bySubject;
}

async function collectUsableFrames(subject, candidates) {
  const usable = [];
  for (const frame of candidates.slice(0, MAX_SCAN_PER_SUBJECT)) {
    if (usable.length >= USABLE_PER_SUBJECT) break;
    const descriptor = await describeFace(path.join(OEP_FRAMES_DIR, frame.frame_id));
    if (descriptor) usable.push(frame);
  }
  console.log(
    `  ${subject}: ${usable.length}/${USABLE_PER_SUBJECT} usable frames found (scanned up to ${Math.min(candidates.length, MAX_SCAN_PER_SUBJECT)})`,
  );
  return usable;
}

function copyFrame(frameId) {
  fs.copyFileSync(path.join(OEP_FRAMES_DIR, frameId), path.join(OUT_FRAMES_DIR, frameId));
}

async function main() {
  if (!fs.existsSync(OEP_FRAMES_DIR)) {
    console.error(`OEP frame pool not found at ${OEP_FRAMES_DIR}.`);
    console.error("This script reuses the raw frame dump from the #1222 face-count eval's Kaggle pipeline");
    console.error("(kept local-only, not committed -- see ../face-detection-eval/kaggle/README.md).");
    process.exit(1);
  }
  fs.mkdirSync(OUT_FRAMES_DIR, { recursive: true });

  await loadModels();

  const manifestRows = readCsv(OEP_MANIFEST);
  const bySubject = groupBySubject(manifestRows);
  const subjects = [...bySubject.keys()].sort();
  console.log(`${subjects.length} subjects in manifest: ${subjects.join(", ")}`);

  const usableBySubject = new Map();
  for (const subject of subjects) {
    usableBySubject.set(subject, await collectUsableFrames(subject, bySubject.get(subject)));
  }

  const enrollmentBySubject = new Map();
  for (const [subject, frames] of usableBySubject) {
    if (frames.length < 2) {
      console.warn(`  skipping ${subject}: only ${frames.length} usable frame(s), need >= 2`);
      continue;
    }
    enrollmentBySubject.set(subject, frames[0]);
  }

  const usedSubjects = [...enrollmentBySubject.keys()];
  const pairs = [];
  let pairId = 0;
  const copiedFrames = new Set();

  const ensureCopied = (frameId) => {
    if (!copiedFrames.has(frameId)) {
      copyFrame(frameId);
      copiedFrames.add(frameId);
    }
  };

  for (const subject of usedSubjects) {
    const enrollment = enrollmentBySubject.get(subject);
    const probes = usableBySubject.get(subject).slice(1);
    ensureCopied(enrollment.frame_id);

    // Genuine pairs: enrollment vs. later same-subject frames.
    for (const probe of probes) {
      ensureCopied(probe.frame_id);
      const deltaS = Math.round(probe.timestamp_s - enrollment.timestamp_s);
      pairs.push({
        pair_id: `pair_${String(++pairId).padStart(4, "0")}`,
        person_id: subject,
        ground_truth_match: 1,
        condition_tags: timeDeltaBucket(deltaS),
        time_delta_s: deltaS,
        enrollment_frame: enrollment.frame_id,
        probe_frame: probe.frame_id,
        source_dataset: SOURCE_DATASET,
        license: LICENSE,
      });
    }

    // Impostor pairs: this subject's enrollment vs. a couple of frames from every other subject.
    for (const otherSubject of usedSubjects) {
      if (otherSubject === subject) continue;
      const otherProbes = usableBySubject.get(otherSubject).slice(1, 1 + IMPOSTOR_PROBES_PER_OTHER_SUBJECT);
      for (const otherProbe of otherProbes) {
        ensureCopied(otherProbe.frame_id);
        pairs.push({
          pair_id: `pair_${String(++pairId).padStart(4, "0")}`,
          person_id: subject,
          ground_truth_match: 0,
          condition_tags: "impostor",
          time_delta_s: "",
          enrollment_frame: enrollment.frame_id,
          probe_frame: otherProbe.frame_id,
          source_dataset: SOURCE_DATASET,
          license: LICENSE,
        });
      }
    }
  }

  writeCsv(
    OUT_PAIRS_CSV,
    [
      "pair_id",
      "person_id",
      "ground_truth_match",
      "condition_tags",
      "time_delta_s",
      "enrollment_frame",
      "probe_frame",
      "source_dataset",
      "license",
    ],
    pairs,
  );

  const genuineCount = pairs.filter((p) => p.ground_truth_match === 1).length;
  const impostorCount = pairs.length - genuineCount;
  console.log(
    `\nWrote ${pairs.length} pairs (${genuineCount} genuine, ${impostorCount} impostor) across ${usedSubjects.length} subjects.`,
  );
  console.log(`Copied ${copiedFrames.size} unique frames into ${OUT_FRAMES_DIR}.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
