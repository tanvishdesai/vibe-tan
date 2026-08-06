// Merges the per-source selection manifests (wider-selection.json,
// mask-selection.json, negatives-selection.json) into the single
// labels.csv the eval script and the test report both read from.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeCsv } from "./lib/csv.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const wider = JSON.parse(
  fs.readFileSync(path.join(__dirname, "wider-selection.json"), "utf8"),
).rows;
const mask = JSON.parse(fs.readFileSync(path.join(__dirname, "mask-selection.json"), "utf8"));
const negatives = JSON.parse(
  fs.readFileSync(path.join(__dirname, "negatives-selection.json"), "utf8"),
);

const rows = [...wider, ...negatives, ...mask];

writeCsv(
  path.join(__dirname, "labels.csv"),
  ["frame_id", "ground_truth_face_count", "condition_tags", "source_dataset", "source_image", "license"],
  rows,
);

console.log(`labels.csv written with ${rows.length} rows`);
