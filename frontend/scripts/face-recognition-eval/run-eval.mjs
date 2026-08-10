// Offline accuracy evaluation for the proctoring face-recognition (identity
// verification) matcher (issue #1224). Runs the *exact* model configuration
// used in production (frontend/src/components/ai/FaceRecognitionComponent.tsx:
// TinyFaceDetector inputSize 512 / scoreThreshold 0.45, FaceLandmark68Net,
// FaceRecognitionNet, Euclidean distance on the 128-d descriptor) against
// every pair in pairs.csv and computes FAR/FRR/EER/ROC, overall and broken
// down by condition tag. Dataset is real exam-webcam footage (MSU OEP,
// reused from the #1222 face-count eval's already-downloaded frame pool --
// see build-pairs.mjs and README.md).
//
// The frontend flags FACE_RECOGNITION when distance >= MATCH_THRESHOLD (0.55,
// hardcoded). This script reproduces that same rule so the metrics mean the
// same thing the production anomaly flag means.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readCsv } from "./lib/csv.mjs";
import { loadModels, describeFace } from "./lib/detect.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRAMES_DIR = path.join(__dirname, "frames");
const QC_DIR = path.join(__dirname, "qc-review");
const PROD_THRESHOLD = 0.55; // FaceRecognitionComponent.tsx MATCH_THRESHOLD
const THRESHOLD_SWEEP = [];
for (let t = 0.3; t <= 0.8 + 1e-9; t += 0.01) THRESHOLD_SWEEP.push(Math.round(t * 100) / 100);

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

function metricsAtThreshold(rows, threshold) {
  // "Match" = distance < threshold (same convention as MATCH_THRESHOLD in prod).
  let genuineTotal = 0,
    genuineAccepted = 0, // correctly matched
    impostorTotal = 0,
    impostorAccepted = 0; // wrongly matched (false accept)
  for (const r of rows) {
    const predictedMatch = r.distance < threshold;
    if (r.ground_truth_match === 1) {
      genuineTotal++;
      if (predictedMatch) genuineAccepted++;
    } else {
      impostorTotal++;
      if (predictedMatch) impostorAccepted++;
    }
  }
  const far = impostorTotal ? impostorAccepted / impostorTotal : null; // False Accept Rate
  const frr = genuineTotal ? 1 - genuineAccepted / genuineTotal : null; // False Reject Rate
  return { threshold, far, frr, genuineTotal, impostorTotal };
}

function findEer(rows) {
  let best = null;
  for (const t of THRESHOLD_SWEEP) {
    const m = metricsAtThreshold(rows, t);
    if (m.far === null || m.frr === null) continue;
    const gap = Math.abs(m.far - m.frr);
    if (!best || gap < best.gap) best = { ...m, gap, eer: (m.far + m.frr) / 2 };
  }
  return best;
}

function rocAuc(rows) {
  // Points sorted by FAR ascending (== threshold ascending here, distance-based).
  const points = THRESHOLD_SWEEP.map((t) => {
    const m = metricsAtThreshold(rows, t);
    return { far: m.far, tar: m.frr === null ? null : 1 - m.frr };
  }).filter((p) => p.far !== null && p.tar !== null);
  points.sort((a, b) => a.far - b.far);
  let auc = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].far - points[i - 1].far;
    const avgY = (points[i].tar + points[i - 1].tar) / 2;
    auc += dx * avgY;
  }
  return { auc, points };
}

function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const r of rows) {
    const key = keyFn(r);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  return groups;
}

async function main() {
  await loadModels();

  const pairsRaw = readCsv(path.join(__dirname, "pairs.csv"));
  console.log(`${pairsRaw.length} pairs loaded from pairs.csv`);

  const uniqueFrames = [...new Set(pairsRaw.flatMap((p) => [p.enrollment_frame, p.probe_frame]))];
  console.log(`Computing descriptors for ${uniqueFrames.length} unique frames...`);
  const descriptors = new Map();
  const start = Date.now();
  for (const frameId of uniqueFrames) {
    const descriptor = await describeFace(path.join(FRAMES_DIR, frameId));
    descriptors.set(frameId, descriptor);
  }
  console.log(`Descriptors computed in ${((Date.now() - start) / 1000).toFixed(1)}s`);

  const rows = [];
  const detectionFailed = [];
  for (const p of pairsRaw) {
    const a = descriptors.get(p.enrollment_frame);
    const b = descriptors.get(p.probe_frame);
    if (!a || !b) {
      detectionFailed.push(p.pair_id);
      continue;
    }
    rows.push({
      pair_id: p.pair_id,
      person_id: p.person_id,
      ground_truth_match: Number(p.ground_truth_match),
      condition_tags: p.condition_tags,
      time_delta_s: p.time_delta_s,
      enrollment_frame: p.enrollment_frame,
      probe_frame: p.probe_frame,
      distance: euclideanDistance(a, b),
    });
  }
  if (detectionFailed.length) {
    console.warn(`${detectionFailed.length} pairs skipped -- re-detection failed on a pre-filtered frame: ${detectionFailed.join(", ")}`);
  }

  const overallAtProd = metricsAtThreshold(rows, PROD_THRESHOLD);
  const eer = findEer(rows);
  const overallAtEer = metricsAtThreshold(rows, eer.threshold);
  const { auc } = rocAuc(rows);

  const byCondition = {};
  for (const [tag, groupRows] of groupBy(
    rows.filter((r) => r.ground_truth_match === 1),
    (r) => r.condition_tags,
  )) {
    byCondition[tag] = {
      atProdThreshold: metricsAtThreshold(groupRows, PROD_THRESHOLD),
      atEerThreshold: metricsAtThreshold(groupRows, eer.threshold),
    };
  }

  const impostorRows = rows.filter((r) => r.ground_truth_match === 0);
  const hardNegatives = [...impostorRows].sort((a, b) => a.distance - b.distance).slice(0, 15);

  const results = {
    productionThreshold: PROD_THRESHOLD,
    totalPairs: rows.length,
    detectionFailedPairs: detectionFailed.length,
    genuinePairs: rows.filter((r) => r.ground_truth_match === 1).length,
    impostorPairs: impostorRows.length,
    subjects: [...new Set(rows.map((r) => r.person_id))].sort(),
    atProductionThreshold: overallAtProd,
    eer: { threshold: eer.threshold, eer: eer.eer, far: eer.far, frr: eer.frr },
    atEerThreshold: overallAtEer,
    rocAuc: auc,
    thresholdSweep: THRESHOLD_SWEEP.map((t) => metricsAtThreshold(rows, t)),
    byCondition,
    hardNegatives: hardNegatives.map((r) => ({ pair_id: r.pair_id, person_id: r.person_id, distance: r.distance, enrollment_frame: r.enrollment_frame, probe_frame: r.probe_frame })),
  };

  fs.writeFileSync(path.join(__dirname, "results.json"), JSON.stringify(results, null, 2));

  // QC review dump: sample genuine pairs per condition bucket + the hard negatives.
  fs.rmSync(QC_DIR, { recursive: true, force: true });
  const dumpPair = (bucket, r) => {
    const dir = path.join(QC_DIR, bucket);
    fs.mkdirSync(dir, { recursive: true });
    const tag = `${r.pair_id}__dist-${r.distance.toFixed(3)}`;
    fs.copyFileSync(path.join(FRAMES_DIR, r.enrollment_frame), path.join(dir, `${tag}__enrollment.jpg`));
    fs.copyFileSync(path.join(FRAMES_DIR, r.probe_frame), path.join(dir, `${tag}__probe.jpg`));
  };
  for (const [tag, groupRows] of groupBy(
    rows.filter((r) => r.ground_truth_match === 1),
    (r) => r.condition_tags,
  )) {
    groupRows.slice(0, 5).forEach((r) => dumpPair(tag, r));
  }
  hardNegatives.slice(0, 10).forEach((r) => dumpPair("hard-negative-impostor", r));

  console.log("\n=== RESULTS ===");
  console.log(`Pairs: ${results.totalPairs} (${results.genuinePairs} genuine, ${results.impostorPairs} impostor), ${results.subjects.length} subjects`);
  console.log(`At production threshold (${PROD_THRESHOLD}): FAR=${(overallAtProd.far * 100).toFixed(2)}% FRR=${(overallAtProd.frr * 100).toFixed(2)}%`);
  console.log(`EER threshold: ${eer.threshold} -> EER=${(eer.eer * 100).toFixed(2)}% (FAR=${(eer.far * 100).toFixed(2)}% FRR=${(eer.frr * 100).toFixed(2)}%)`);
  console.log(`ROC AUC: ${auc.toFixed(4)}`);
  console.log("\nBy condition (genuine pairs, FRR):");
  for (const [tag, m] of Object.entries(byCondition)) {
    console.log(`  ${tag}: FRR@prod=${(m.atProdThreshold.frr * 100).toFixed(1)}% (n=${m.atProdThreshold.genuineTotal}) FRR@eer=${(m.atEerThreshold.frr * 100).toFixed(1)}%`);
  }
  console.log(`\nHardest impostor pairs (closest distance):`);
  hardNegatives.slice(0, 5).forEach((r) => console.log(`  ${r.pair_id} (${r.person_id} vs ${r.probe_frame}): distance=${r.distance.toFixed(3)}`));
  console.log(`\nWrote results.json and qc-review/`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
