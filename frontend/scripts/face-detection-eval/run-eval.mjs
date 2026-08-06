// Offline accuracy evaluation for the proctoring face-count detector
// (issue #1222). Runs the *exact* model configuration used in production
// (frontend/src/components/ai/FaceDetectorWorker.ts: MediaPipeFaceDetector,
// runtime "tfjs", modelType "full", maxFaces 10) against every frame in
// labels.csv and computes precision/recall/F1 + a confusion matrix, overall
// and broken down by condition tag.
//
// The frontend maps raw face count to proctoring anomalies as:
//   0 faces  -> NO_FACE
//   1 face   -> OK (no anomaly)
//   2+ faces -> MULTIPLE_FACES
// (see frontend/src/components/floating-video.tsx handleImageAnomaly). This
// script reproduces that same bucketing so the metrics mean the same thing
// the production anomaly flags mean.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import { readCsv } from "./lib/csv.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRAMES_DIR = path.join(__dirname, "frames");
const MAX_SIDE = 1280; // cap resolution to a realistic webcam-frame scale

function bucketOf(count) {
  if (count === 0) return "NO_FACE";
  if (count === 1) return "OK";
  return "MULTIPLE_FACES";
}

function decodeToTensor(tf, filePath) {
  const raw = jpeg.decode(fs.readFileSync(filePath), { useTArray: true });
  let { data, width, height } = raw;

  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  if (scale < 1) {
    const newW = Math.round(width * scale);
    const newH = Math.round(height * scale);
    const resized = new Uint8Array(newW * newH * 4);
    for (let y = 0; y < newH; y++) {
      const srcY = Math.min(height - 1, Math.round(y / scale));
      for (let x = 0; x < newW; x++) {
        const srcX = Math.min(width - 1, Math.round(x / scale));
        const srcIdx = (srcY * width + srcX) * 4;
        const dstIdx = (y * newW + x) * 4;
        resized[dstIdx] = data[srcIdx];
        resized[dstIdx + 1] = data[srcIdx + 1];
        resized[dstIdx + 2] = data[srcIdx + 2];
        resized[dstIdx + 3] = data[srcIdx + 3];
      }
    }
    data = resized;
    width = newW;
    height = newH;
  }

  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i];
    rgb[j + 1] = data[i + 1];
    rgb[j + 2] = data[i + 2];
  }
  return tf.tensor3d(rgb, [height, width, 3], "int32");
}

function computeBinaryMetrics(rows, positiveBucket, predKey) {
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  for (const r of rows) {
    const actual = r.groundTruthBucket === positiveBucket;
    const predicted = r[predKey] === positiveBucket;
    if (actual && predicted) tp++;
    else if (!actual && predicted) fp++;
    else if (actual && !predicted) fn++;
    else tn++;
  }
  const precision = tp + fp === 0 ? null : tp / (tp + fp);
  const recall = tp + fn === 0 ? null : tp / (tp + fn);
  const f1 =
    precision === null || recall === null || precision + recall === 0
      ? null
      : (2 * precision * recall) / (precision + recall);
  return { tp, fp, fn, tn, precision, recall, f1 };
}

function confusionMatrix(rows, predKey) {
  const buckets = ["NO_FACE", "OK", "MULTIPLE_FACES"];
  const matrix = Object.fromEntries(
    buckets.map((a) => [a, Object.fromEntries(buckets.map((p) => [p, 0]))]),
  );
  for (const r of rows) {
    matrix[r.groundTruthBucket][r[predKey]]++;
  }
  return matrix;
}

async function main() {
  const tf = await import("@tensorflow/tfjs-core");
  await import("@tensorflow/tfjs-backend-wasm");
  const faceDetection = await import("@tensorflow-models/face-detection");

  await tf.setBackend("wasm");
  await tf.ready();

  const detector = await faceDetection.createDetector(
    faceDetection.SupportedModels.MediaPipeFaceDetector,
    { runtime: "tfjs", maxFaces: 10, modelType: "full" },
  );

  const labels = readCsv(path.join(__dirname, "labels.csv"));
  console.log(`Evaluating ${labels.length} frames...\n`);

  const rows = [];
  let processed = 0;
  for (const label of labels) {
    const framePath = path.join(FRAMES_DIR, label.frame_id);
    if (!fs.existsSync(framePath)) {
      console.warn(`  missing frame, skipping: ${label.frame_id}`);
      continue;
    }
    let tensor;
    try {
      tensor = decodeToTensor(tf, framePath);
      const faces = await detector.estimateFaces(tensor);
      const predictedCount = faces.length;
      const groundTruthCount = Number(label.ground_truth_face_count);
      rows.push({
        frame_id: label.frame_id,
        condition_tags: label.condition_tags,
        source: label.frame_id.startsWith("oep_") ? "OEP (real exam webcam)" : "WIDER_FACE / mask dataset",
        groundTruthCount,
        groundTruthBucket: bucketOf(groundTruthCount),
        predictedCount,
        predictedBucket: bucketOf(predictedCount),
      });
    } catch (err) {
      console.warn(`  detection failed on ${label.frame_id}: ${err.message}`);
    } finally {
      tensor?.dispose();
    }
    processed++;
    if (processed % 40 === 0) console.log(`  ${processed}/${labels.length}`);
  }

  const overallAccuracy =
    rows.filter((r) => r.predictedBucket === r.groundTruthBucket).length / rows.length;

  const noFaceMetrics = computeBinaryMetrics(rows, "NO_FACE", "predictedBucket");
  const multiFaceMetrics = computeBinaryMetrics(rows, "MULTIPLE_FACES", "predictedBucket");
  const overallConfusion = confusionMatrix(rows, "predictedBucket");

  const conditionBreakdown = {};
  for (const tag of new Set(rows.map((r) => r.condition_tags))) {
    const subset = rows.filter((r) => r.condition_tags === tag);
    conditionBreakdown[tag] = {
      frameCount: subset.length,
      accuracy: subset.filter((r) => r.predictedBucket === r.groundTruthBucket).length / subset.length,
      noFace: computeBinaryMetrics(subset, "NO_FACE", "predictedBucket"),
      multipleFaces: computeBinaryMetrics(subset, "MULTIPLE_FACES", "predictedBucket"),
      confusion: confusionMatrix(subset, "predictedBucket"),
    };
  }

  const sourceBreakdown = {};
  for (const source of new Set(rows.map((r) => r.source))) {
    const subset = rows.filter((r) => r.source === source);
    sourceBreakdown[source] = {
      frameCount: subset.length,
      accuracy: subset.filter((r) => r.predictedBucket === r.groundTruthBucket).length / subset.length,
      noFace: computeBinaryMetrics(subset, "NO_FACE", "predictedBucket"),
      multipleFaces: computeBinaryMetrics(subset, "MULTIPLE_FACES", "predictedBucket"),
      confusion: confusionMatrix(subset, "predictedBucket"),
    };
  }

  const misclassified = rows
    .filter((r) => r.predictedBucket !== r.groundTruthBucket)
    .map((r) => ({
      frame_id: r.frame_id,
      condition_tags: r.condition_tags,
      groundTruth: `${r.groundTruthBucket} (${r.groundTruthCount})`,
      predicted: `${r.predictedBucket} (${r.predictedCount})`,
    }));

  const results = {
    modelConfig: { runtime: "tfjs", modelType: "full", maxFaces: 10 },
    frameCount: rows.length,
    overallAccuracy,
    overallConfusion,
    noFaceMetrics,
    multiFaceMetrics,
    sourceBreakdown,
    conditionBreakdown,
    misclassified,
  };

  fs.writeFileSync(path.join(__dirname, "results.json"), JSON.stringify(results, null, 2));

  console.log("\n=== Overall ===");
  console.log(`Frames evaluated: ${rows.length}`);
  console.log(`Overall bucket accuracy: ${(overallAccuracy * 100).toFixed(1)}%`);
  console.log(
    `NO_FACE        precision=${fmt(noFaceMetrics.precision)} recall=${fmt(noFaceMetrics.recall)} f1=${fmt(noFaceMetrics.f1)} (tp=${noFaceMetrics.tp} fp=${noFaceMetrics.fp} fn=${noFaceMetrics.fn})`,
  );
  console.log(
    `MULTIPLE_FACES precision=${fmt(multiFaceMetrics.precision)} recall=${fmt(multiFaceMetrics.recall)} f1=${fmt(multiFaceMetrics.f1)} (tp=${multiFaceMetrics.tp} fp=${multiFaceMetrics.fp} fn=${multiFaceMetrics.fn})`,
  );

  console.log("\n=== Confusion matrix (rows=ground truth, cols=predicted) ===");
  console.table(overallConfusion);

  console.log("\n=== By source ===");
  for (const [source, m] of Object.entries(sourceBreakdown)) {
    console.log(
      `${source.padEnd(28)} n=${String(m.frameCount).padEnd(4)} acc=${(m.accuracy * 100).toFixed(1)}%  NO_FACE f1=${fmt(m.noFace.f1)}  MULTIPLE_FACES f1=${fmt(m.multipleFaces.f1)}`,
    );
  }

  console.log("\n=== By condition ===");
  for (const [tag, m] of Object.entries(conditionBreakdown)) {
    console.log(
      `${tag.padEnd(20)} n=${String(m.frameCount).padEnd(4)} acc=${(m.accuracy * 100).toFixed(1)}%  NO_FACE f1=${fmt(m.noFace.f1)}  MULTIPLE_FACES f1=${fmt(m.multipleFaces.f1)}`,
    );
  }

  console.log(`\nFull results written to results.json`);
}

function fmt(v) {
  return v === null ? "n/a" : v.toFixed(3);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
