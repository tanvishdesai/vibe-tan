// Builds the "0 faces" slice of the labeled test set.
//
// WIDER FACE has no true negatives (every image was selected because it
// contains a face), so instead of sourcing a whole new dataset we derive
// negatives the standard hard-negative-mining way: crop a face-free region
// out of a WIDER FACE image, verified programmatically against that image's
// own ground-truth boxes (valid + invalid) so the crop is guaranteed to
// contain zero faces.
//
// Usage:
//   node build-dataset-negatives.mjs <WIDER_val/images dir> <wider_face_val_bbx_gt.txt>
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import { parseWiderAnnotations, seededShuffle } from "./lib/wider-face.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRAMES_DIR = path.join(__dirname, "frames");
const SEED = 20260806;
const TARGET_COUNT = 40;
const CROP_SIZE = 320; // roughly a webcam-frame-scale region
const MARGIN = 24; // px kept clear around every annotated box (valid or not)

const [, , imagesDir, annotationFile] = process.argv;
if (!imagesDir || !annotationFile) {
  console.error(
    "Usage: node build-dataset-negatives.mjs <WIDER_val/images dir> <wider_face_val_bbx_gt.txt>",
  );
  process.exit(1);
}

const usedSourceFiles = new Set(
  JSON.parse(fs.readFileSync(path.join(__dirname, "wider-selection.json"), "utf8"))
    .usedSourceFiles,
);

const images = parseWiderAnnotations(annotationFile).filter(
  (img) => !usedSourceFiles.has(img.file) && img.boxes.length > 0,
);
const candidates = seededShuffle(images, SEED);

function boxesOverlapCandidate(cx, cy, size, boxes) {
  return boxes.some((b) => {
    const bx0 = b.x - MARGIN;
    const by0 = b.y - MARGIN;
    const bx1 = b.x + b.w + MARGIN;
    const by1 = b.y + b.h + MARGIN;
    return !(cx + size < bx0 || cx > bx1 || cy + size < by0 || cy > by1);
  });
}

function findFaceFreeCrop(width, height, boxes) {
  const size = Math.min(CROP_SIZE, width, height);
  if (size < 100) return null;
  const stepsX = Math.max(1, Math.floor((width - size) / 40));
  const stepsY = Math.max(1, Math.floor((height - size) / 40));
  for (let sy = 0; sy <= stepsY; sy++) {
    for (let sx = 0; sx <= stepsX; sx++) {
      const cx = sx * 40;
      const cy = sy * 40;
      if (cx + size > width || cy + size > height) continue;
      if (!boxesOverlapCandidate(cx, cy, size, boxes)) {
        return { x: cx, y: cy, size };
      }
    }
  }
  return null;
}

fs.mkdirSync(FRAMES_DIR, { recursive: true });
const rows = [];
let seq = 1;

for (const img of candidates) {
  if (rows.length >= TARGET_COUNT) break;

  let decoded;
  try {
    decoded = jpeg.decode(fs.readFileSync(path.join(imagesDir, img.file)), { useTArray: true });
  } catch {
    continue; // a handful of WIDER images are malformed/non-baseline JPEGs; skip them
  }
  const { width, height, data } = decoded;
  const crop = findFaceFreeCrop(width, height, img.boxes);
  if (!crop) continue;

  const out = Buffer.alloc(crop.size * crop.size * 4);
  const rowBytes = crop.size * 4;
  for (let row = 0; row < crop.size; row++) {
    const srcStart = ((crop.y + row) * width + crop.x) * 4;
    const dstStart = row * rowBytes;
    for (let k = 0; k < rowBytes; k++) {
      out[dstStart + k] = data[srcStart + k];
    }
  }
  const encoded = jpeg.encode({ data: out, width: crop.size, height: crop.size }, 90);

  const frameId = `neg_${String(seq).padStart(4, "0")}.jpg`;
  seq++;
  fs.writeFileSync(path.join(FRAMES_DIR, frameId), encoded.data);
  rows.push({
    frame_id: frameId,
    ground_truth_face_count: 0,
    condition_tags: "background_crop",
    source_dataset: "WIDER_FACE_val (derived negative)",
    source_image: `${img.file} @ (${crop.x},${crop.y},${crop.size})`,
    license: "CC-BY-NC-ND-4.0",
  });
}

fs.writeFileSync(path.join(__dirname, "negatives-selection.json"), JSON.stringify(rows, null, 2));
console.log(`Wrote ${rows.length}/${TARGET_COUNT} negative (0-face) frames`);
