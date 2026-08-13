import type { ExpertScore } from "./shared.js";

const SIZE = 256;
const WAVELET_SIZE = 130;
const INV_SQRT_2 = Math.SQRT1_2;

let sessionPromise: Promise<any> | undefined;

async function getSession(ort: any) {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(
      chrome.runtime.getURL("models/safe/model.onnx"),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
    );
  }
  return sessionPromise;
}

export async function detectSafe(bitmap: ImageBitmap): Promise<ExpertScore> {
  const ort = await import(chrome.runtime.getURL("vendor/ort.all.min.mjs"));
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
  ort.env.wasm.numThreads = 1;
  const session = await getSession(ort);
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas unavailable");
  const sourceX = Math.max(0, Math.floor((bitmap.width - SIZE) / 2));
  const sourceY = Math.max(0, Math.floor((bitmap.height - SIZE) / 2));
  const width = Math.min(bitmap.width, SIZE), height = Math.min(bitmap.height, SIZE);
  context.drawImage(bitmap, sourceX, sourceY, width, height, Math.max(0, Math.floor((SIZE - bitmap.width) / 2)), Math.max(0, Math.floor((SIZE - bitmap.height) / 2)), width, height);
  const band = safeDiagonalBand(context.getImageData(0, 0, SIZE, SIZE).data);
  const output = await session.run({ diagonal_band: new ort.Tensor("float32", band, [1, 3, SIZE, SIZE]) });
  const realLogit = Number(output.logits.data[0]), fakeLogit = Number(output.logits.data[1]);
  const score = 1 / (1 + Math.exp(realLogit - fakeLogit));
  return { source: "SAFE", score, active: false, detail: `wavelet forensic ${(score * 100).toFixed(1)}% (shadow mode pending calibration)` };
}

export function safeDiagonalBand(rgba: Uint8ClampedArray | Uint8Array): Float32Array {
  const horizontal = new Float32Array(3 * SIZE * WAVELET_SIZE);
  for (let channel = 0; channel < 3; channel++) {
    const offset = channel * SIZE * WAVELET_SIZE;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < WAVELET_SIZE; x++) {
        const left = symmetric(x * 2 - 2), right = symmetric(x * 2 - 1);
        horizontal[offset + y * WAVELET_SIZE + x] = INV_SQRT_2 * (
          rgba[(y * SIZE + left) * 4 + channel] - rgba[(y * SIZE + right) * 4 + channel]
        ) / 255;
      }
    }
  }

  const wavelet = new Float32Array(3 * WAVELET_SIZE * WAVELET_SIZE);
  for (let channel = 0; channel < 3; channel++) {
    const inputOffset = channel * SIZE * WAVELET_SIZE;
    const outputOffset = channel * WAVELET_SIZE * WAVELET_SIZE;
    for (let y = 0; y < WAVELET_SIZE; y++) {
      const upper = symmetric(y * 2 - 2), lower = symmetric(y * 2 - 1);
      for (let x = 0; x < WAVELET_SIZE; x++) {
        wavelet[outputOffset + y * WAVELET_SIZE + x] = INV_SQRT_2 * (
          horizontal[inputOffset + upper * WAVELET_SIZE + x] - horizontal[inputOffset + lower * WAVELET_SIZE + x]
        );
      }
    }
  }
  return bilinearResize(wavelet);
}

function bilinearResize(input: Float32Array): Float32Array {
  const output = new Float32Array(3 * SIZE * SIZE);
  const scale = WAVELET_SIZE / SIZE;
  for (let channel = 0; channel < 3; channel++) {
    const inputOffset = channel * WAVELET_SIZE * WAVELET_SIZE;
    const outputOffset = channel * SIZE * SIZE;
    for (let y = 0; y < SIZE; y++) {
      const sourceY = Math.max(0, Math.min(WAVELET_SIZE - 1, (y + 0.5) * scale - 0.5));
      const y0 = Math.floor(sourceY), y1 = Math.min(WAVELET_SIZE - 1, y0 + 1), fy = sourceY - y0;
      for (let x = 0; x < SIZE; x++) {
        const sourceX = Math.max(0, Math.min(WAVELET_SIZE - 1, (x + 0.5) * scale - 0.5));
        const x0 = Math.floor(sourceX), x1 = Math.min(WAVELET_SIZE - 1, x0 + 1), fx = sourceX - x0;
        const top = input[inputOffset + y0 * WAVELET_SIZE + x0] * (1 - fx) + input[inputOffset + y0 * WAVELET_SIZE + x1] * fx;
        const bottom = input[inputOffset + y1 * WAVELET_SIZE + x0] * (1 - fx) + input[inputOffset + y1 * WAVELET_SIZE + x1] * fx;
        output[outputOffset + y * SIZE + x] = top * (1 - fy) + bottom * fy;
      }
    }
  }
  return output;
}

function symmetric(index: number): number {
  if (index < 0) return -index - 1;
  if (index >= SIZE) return SIZE * 2 - index - 1;
  return index;
}
