// Builds the 1-face / multi-face / single-attribute-edge-case slices of the
// labeled test set from the official WIDER FACE validation split.
//
// WIDER FACE ships real per-face attribute annotations (blur, illumination,
// occlusion, pose) which map directly onto the condition tags the eval needs,
// so ground truth here comes from the dataset's own labels rather than manual
// re-labeling.
//
// Usage:
//   node build-dataset-wider.mjs <path-to-WIDER_val/images> <path-to-wider_face_val_bbx_gt.txt>
//
// Source: https://huggingface.co/datasets/CUHK-CSE/wider_face
// License: CC BY-NC-ND 4.0 (non-commercial research/testing use)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseWiderAnnotations, realFaces, seededShuffle } from "./lib/wider-face.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRAMES_DIR = path.join(__dirname, "frames");
const SEED = 20260806; // fixed so re-running reproduces the same selection

const [, , imagesDir, annotationFile] = process.argv;
if (!imagesDir || !annotationFile) {
  console.error(
    "Usage: node build-dataset-wider.mjs <WIDER_val/images dir> <wider_face_val_bbx_gt.txt>",
  );
  process.exit(1);
}

const images = parseWiderAnnotations(annotationFile).map((img) => ({
  ...img,
  faces: realFaces(img.boxes),
}));

function pick(predicate, count) {
  const matches = images.filter(predicate);
  return seededShuffle(matches, SEED).slice(0, count);
}

// Each bucket: [selection predicate, how many frames, ground-truth count producer, condition tags]
const buckets = [
  {
    name: "one_face_clean",
    tags: "normal",
    count: 70,
    select: pick(
      (img) =>
        img.faces.length === 1 &&
        img.faces[0].blur === 0 &&
        img.faces[0].illumination === 0 &&
        img.faces[0].occlusion === 0 &&
        img.faces[0].pose === 0,
      70,
    ),
  },
  {
    name: "multiple_faces",
    tags: "peer_present",
    count: 55,
    select: pick((img) => img.faces.length >= 2 && img.faces.length <= 4, 55),
  },
  {
    name: "occlusion_heavy",
    tags: "occlusion",
    count: 20,
    select: pick((img) => img.faces.length === 1 && img.faces[0].occlusion === 2, 20),
  },
  {
    name: "occlusion_partial",
    tags: "partial_occlusion",
    count: 15,
    select: pick((img) => img.faces.length === 1 && img.faces[0].occlusion === 1, 15),
  },
  {
    name: "pose_atypical",
    tags: "angle",
    count: 20,
    select: pick(
      (img) => img.faces.length === 1 && img.faces[0].pose === 1 && img.faces[0].occlusion === 0,
      20,
    ),
  },
  {
    name: "illumination_extreme",
    tags: "lighting",
    count: 20,
    select: pick(
      (img) =>
        img.faces.length === 1 && img.faces[0].illumination === 1 && img.faces[0].occlusion === 0,
      20,
    ),
  },
  {
    name: "blur_heavy",
    tags: "blur",
    count: 15,
    select: pick(
      (img) => img.faces.length === 1 && img.faces[0].blur === 2 && img.faces[0].occlusion === 0,
      15,
    ),
  },
];

fs.mkdirSync(FRAMES_DIR, { recursive: true });

const rows = [];
let frameSeq = 1;
const usedSourceFiles = new Set();

for (const bucket of buckets) {
  for (const img of bucket.select) {
    const ext = path.extname(img.file) || ".jpg";
    const frameId = `wf_${String(frameSeq).padStart(4, "0")}${ext}`;
    frameSeq++;
    fs.copyFileSync(path.join(imagesDir, img.file), path.join(FRAMES_DIR, frameId));
    usedSourceFiles.add(img.file);
    rows.push({
      frame_id: frameId,
      ground_truth_face_count: img.faces.length,
      condition_tags: bucket.tags,
      source_dataset: "WIDER_FACE_val",
      source_image: img.file,
      license: "CC-BY-NC-ND-4.0",
    });
  }
  console.log(`${bucket.name}: selected ${bucket.select.length}/${bucket.count} requested`);
}

fs.writeFileSync(
  path.join(__dirname, "wider-selection.json"),
  JSON.stringify({ rows, usedSourceFiles: [...usedSourceFiles] }, null, 2),
);

console.log(`\nTotal WIDER FACE frames written: ${rows.length}`);
