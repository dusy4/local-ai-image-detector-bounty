import type { ExpertScore } from "./shared.js";

const SIZE = 256;
const MEAN = [0.48145466, 0.4578275, 0.40821073];
const STD = [0.26862954, 0.26130258, 0.27577711];

let runtimePromise: Promise<any> | undefined;
let sessionPromise: Promise<any> | undefined;

async function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = import(chrome.runtime.getURL("vendor/ort.all.min.mjs")).then((ort: any) => {
      ort.env.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
      ort.env.wasm.numThreads = 1;
      return ort;
    });
  }
  return runtimePromise;
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = getRuntime().then((ort: any) => ort.InferenceSession.create(
      chrome.runtime.getURL("models/ferretnet/model.onnx"),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
    ));
  }
  return sessionPromise;
}

export async function detectFerretNet(bitmap: ImageBitmap): Promise<ExpertScore> {
  const [ort, session] = await Promise.all([getRuntime(), getSession()]);
  const residual = preprocess(bitmap);
  const output = await session.run({ residual: new ort.Tensor("float32", residual, [1, 3, SIZE, SIZE]) });
  const logit = Number(output.logit.data[0]);
  const score = logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));
  return {
    source: "FerretNet",
    score,
    active: false,
    detail: `local-pixel residual ${(score * 100).toFixed(1)}% (shadow mode pending calibration)`,
  };
}

function preprocess(bitmap: ImageBitmap): Float32Array {
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas unavailable");
  context.imageSmoothingEnabled = false;
  const sourceX = Math.max(0, Math.floor((bitmap.width - SIZE) / 2));
  const sourceY = Math.max(0, Math.floor((bitmap.height - SIZE) / 2));
  const width = Math.min(bitmap.width, SIZE);
  const height = Math.min(bitmap.height, SIZE);
  const destinationX = Math.max(0, Math.floor((SIZE - bitmap.width) / 2));
  const destinationY = Math.max(0, Math.floor((SIZE - bitmap.height) / 2));
  context.drawImage(bitmap, sourceX, sourceY, width, height, destinationX, destinationY, width, height);
  const rgba = context.getImageData(0, 0, SIZE, SIZE).data;
  const normalized = new Float32Array(3 * SIZE * SIZE);
  for (let channel = 0; channel < 3; channel++) {
    const offset = channel * SIZE * SIZE;
    for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
      normalized[offset + pixel] = (rgba[pixel * 4 + channel] / 255 - MEAN[channel]) / STD[channel];
    }
  }

  const residual = new Float32Array(normalized.length);
  const neighborhood = new Array<number>(9);
  for (let channel = 0; channel < 3; channel++) {
    const offset = channel * SIZE * SIZE;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx, ny = y + dy;
            neighborhood[count++] = (dx === 0 && dy === 0) || nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE
              ? 0
              : normalized[offset + ny * SIZE + nx];
          }
        }
        for (let index = 1; index < 9; index++) {
          const value = neighborhood[index];
          let cursor = index - 1;
          while (cursor >= 0 && neighborhood[cursor] > value) {
            neighborhood[cursor + 1] = neighborhood[cursor--];
          }
          neighborhood[cursor + 1] = value;
        }
        const pixel = y * SIZE + x;
        residual[offset + pixel] = normalized[offset + pixel] - neighborhood[4];
      }
    }
  }
  return residual;
}
