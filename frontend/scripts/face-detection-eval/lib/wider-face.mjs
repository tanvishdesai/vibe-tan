// Parser for the official WIDER FACE ground-truth annotation format
// (wider_face_split/wider_face_val_bbx_gt.txt). One entry per line group:
//   <relative image path>
//   <number of boxes>
//   <x> <y> <w> <h> <blur> <expression> <illumination> <invalid> <occlusion> <pose>   (repeated per box)
// Images with zero faces still carry a single dummy all-zero box line.
import fs from "node:fs";

export function parseWiderAnnotations(txtPath) {
  const lines = fs.readFileSync(txtPath, "utf8").split("\n");
  const images = [];
  let i = 0;
  while (i < lines.length) {
    const file = lines[i]?.trim();
    i++;
    if (!file) continue;

    const numBoxes = parseInt(lines[i]?.trim(), 10);
    i++;
    const linesToRead = Math.max(numBoxes, 1);
    const boxes = [];
    for (let k = 0; k < linesToRead; k++) {
      const parts = lines[i]?.trim().split(/\s+/).map(Number);
      i++;
      if (numBoxes === 0 || !parts || parts.length < 10) continue;
      const [x, y, w, h, blur, expression, illumination, invalid, occlusion, pose] = parts;
      boxes.push({ x, y, w, h, blur, expression, illumination, invalid, occlusion, pose });
    }
    images.push({ file, boxes });
  }
  return images;
}

// Faces too small to matter for a single/few-person webcam proctoring frame
// (WIDER FACE includes dense-crowd shots with faces only a few px wide).
const MIN_FACE_SIDE_PX = 50;

export function realFaces(boxes) {
  return boxes.filter(
    (b) => b.invalid === 0 && b.w >= MIN_FACE_SIDE_PX && b.h >= MIN_FACE_SIDE_PX,
  );
}

// Small deterministic PRNG (mulberry32) so dataset selection is reproducible
// without needing to vendor/ship the exact chosen file list separately.
export function seededShuffle(array, seed) {
  let a = seed;
  const rand = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = array.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
