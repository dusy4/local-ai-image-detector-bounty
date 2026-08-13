import type { ExpertScore } from "./shared.js";

const SIZE = 256;
const RESIZE = 288;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
let sessionPromise: Promise<any> | undefined;

async function getSession(ort: any) {
  return sessionPromise ??= ort.InferenceSession.create(
    chrome.runtime.getURL("models/modern-convnext/model.onnx"),
    { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
  );
}

export async function detectModernConvNext(bitmap: ImageBitmap): Promise<ExpertScore> {
  const ort = await import(chrome.runtime.getURL("vendor/ort.all.min.mjs"));
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
  ort.env.wasm.numThreads = 1;
  const session = await getSession(ort);
  const canvas = new OffscreenCanvas(RESIZE, RESIZE);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas unavailable");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  const scale = RESIZE / Math.min(bitmap.width, bitmap.height);
  const width = bitmap.width * scale, height = bitmap.height * scale;
  context.drawImage(bitmap, (RESIZE - width) / 2, (RESIZE - height) / 2, width, height);
  const rgba = context.getImageData(16, 16, SIZE, SIZE).data;
  const input = new Float32Array(3 * SIZE * SIZE);
  for (let channel = 0; channel < 3; channel++) {
    const offset = channel * SIZE * SIZE;
    for (let pixel = 0; pixel < SIZE * SIZE; pixel++) input[offset + pixel] = (rgba[pixel * 4 + channel] / 255 - MEAN[channel]) / STD[channel];
  }
  const output = await session.run({ pixel_values: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]) });
  const logits = output.logits.data;
  const maximum = Math.max(Number(logits[0]), Number(logits[1]));
  const real = Math.exp(Number(logits[0]) - maximum), fake = Math.exp(Number(logits[1]) - maximum);
  const score = fake / (real + fake);
  return { source: "Modern ConvNeXt", score, active: true, detail: `current-generator expert ${(score * 100).toFixed(1)}% (active; cross-dataset calibrated)` };
}
