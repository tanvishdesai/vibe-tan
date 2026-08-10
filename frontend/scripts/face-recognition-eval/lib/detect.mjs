// Shared face-api.js Node setup: the exact detector config used in production
// (frontend/src/components/ai/FaceRecognitionComponent.tsx: TinyFaceDetector,
// inputSize 512, scoreThreshold 0.45, FaceLandmark68Net, FaceRecognitionNet),
// running against raw tf.tensor3d input under the WASM backend -- no DOM/canvas.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import * as faceapi from "@vladmandic/face-api/dist/face-api.node-wasm.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../node_modules/@vladmandic/face-api/model");
const MAX_SIDE = 1280; // cap resolution to a realistic webcam-frame scale, same as face-detection-eval

export { faceapi };

export async function loadModels() {
  await faceapi.tf.setBackend("wasm");
  await faceapi.tf.ready();
  await faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_DIR);
}

export function decodeToTensor(filePath) {
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
  return faceapi.tf.tensor3d(rgb, [height, width, 3], "int32");
}

// Returns the 128-d descriptor for the frame's face, or null if the frame
// doesn't have *exactly one* face above the production score threshold.
// Deliberately not detectSingleFace(): that just returns the single
// highest-scoring face, which silently picks a different person when a
// bystander is also in frame (OEP frames sometimes have two people) --
// verified this was corrupting genuine pairs during dataset construction
// (two frames of the same enrolled subject, 2s apart, where the detector's
// top-scoring face flipped to a different person because their score
// happened to edge out the subject's in one frame). Requiring exactly one
// candidate face is also the more faithful match for what the production
// component actually expects: one person alone at their desk.
export async function describeFace(filePath) {
  const tensor = decodeToTensor(filePath);
  try {
    const results = await faceapi
      .detectAllFaces(tensor, new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.45 }))
      .withFaceLandmarks()
      .withFaceDescriptors();
    if (results.length !== 1) return null;
    return Array.from(results[0].descriptor);
  } finally {
    tensor.dispose();
  }
}
