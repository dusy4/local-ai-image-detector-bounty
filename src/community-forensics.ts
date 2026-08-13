import type { ExpertScore } from "./shared.js";

const SIZE = 224;
const RESIZE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
let sessionPromise: Promise<any> | undefined;

async function getSession(ort: any) {
  return sessionPromise ??= ort.InferenceSession.create(
    chrome.runtime.getURL("models/community-forensics-official/model.onnx"),
    { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
  );
}

export async function detectCommunityForensics(bitmap: ImageBitmap, multiCrop = false): Promise<ExpertScore> {
  const ort = await import(chrome.runtime.getURL("vendor/ort.all.min.mjs"));
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
  ort.env.wasm.numThreads = 1;
  const session = await getSession(ort);
  const inputs = makeInputs(bitmap, multiCrop);
  let score = 0;
  for (const input of inputs) {
    const output = await session.run({ pixel_values: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]) });
    const logit = Number(output.logit.data[0]);
    score = Math.max(score, logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit)));
  }
  return { source: "Community Forensics", score, active: true, detail: `broad generator expert ${(score * 100).toFixed(1)}% (${multiCrop ? "selective max-6 TTA" : "center crop"}; active)` };
}

function makeInputs(bitmap: ImageBitmap, multiCrop: boolean): Float32Array[] {
  const specs: Array<[number, "center" | "tl" | "tr" | "bl" | "br"]> = [[RESIZE, "center"]];
  if (multiCrop) specs.push([RESIZE, "tl"], [RESIZE, "tr"], [RESIZE, "bl"], [RESIZE, "br"], [288, "center"]);
  return specs.map(([shortEdge, origin]) => {
    const scale = shortEdge / Math.min(bitmap.width, bitmap.height);
    const width = Math.round(bitmap.width * scale), height = Math.round(bitmap.height * scale);
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas unavailable");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, width, height);
    const x = origin === "center" ? (width - SIZE) / 2 : origin.endsWith("r") ? width - SIZE : 0;
    const y = origin === "center" ? (height - SIZE) / 2 : origin.startsWith("b") ? height - SIZE : 0;
    const rgba = context.getImageData(x, y, SIZE, SIZE).data;
    const input = new Float32Array(3 * SIZE * SIZE);
    for (let channel = 0; channel < 3; channel++) {
      const offset = channel * SIZE * SIZE;
      for (let pixel = 0; pixel < SIZE * SIZE; pixel++) input[offset + pixel] = (rgba[pixel * 4 + channel] / 255 - MEAN[channel]) / STD[channel];
    }
    return input;
  });
}
