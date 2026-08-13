import type { Signal } from "./shared.js";

type CarrierFile = {
  version: string;
  imageSize: number;
  sets: Array<{ name: string; carriers: Array<[number, number]>; phases: number[] }>;
  calibration: { center: number; steepness: number; minimumReportScore: number };
};

let carrierPromise: Promise<CarrierFile | null> | undefined;

export async function detectSynthId(bitmap: ImageBitmap): Promise<Signal | null> {
  const config = await loadCarriers();
  if (!config?.sets.some(set => set.phases.length === set.carriers.length)) return null;
  const pixels = resizeGray(bitmap, config.imageSize);
  let best = 0;
  let bestName = "";
  for (const set of config.sets) {
    if (set.phases.length !== set.carriers.length) continue;
    const phases = carrierPhases(pixels, config.imageSize, set.carriers);
    const matches = phases.map((phase, i) => 1 - angularDistance(phase, set.phases[i]) / Math.PI);
    const mean = matches.reduce((sum, value) => sum + value, 0) / matches.length;
    if (mean > best) { best = mean; bestName = set.name; }
  }
  const score = 1 / (1 + Math.exp(-config.calibration.steepness * (best - config.calibration.center)));
  if (score < config.calibration.minimumReportScore) return null;
  return { source: "SynthID (experimental public decoder)", score, detail: `${bestName} carrier phase match ${best.toFixed(3)}` };
}

function loadCarriers(): Promise<CarrierFile | null> {
  carrierPromise ??= fetch(chrome.runtime.getURL("synthid-carriers.json"))
    .then(response => response.ok ? response.json() : null)
    .catch(() => null);
  return carrierPromise;
}

function resizeGray(bitmap: ImageBitmap, size: number): Float32Array {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const gray = new Float32Array(size * size);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) gray[i] = (rgba[p] + rgba[p + 1] + rgba[p + 2]) / 3;
  return gray;
}

function carrierPhases(pixels: Float32Array, size: number, carriers: Array<[number, number]>): number[] {
  const uniqueFx = [...new Set(carriers.map(([, fx]) => fx))];
  const rowProjections = new Map<number, { re: Float64Array; im: Float64Array }>();
  for (const fx of uniqueFx) {
    const re = new Float64Array(size); const im = new Float64Array(size);
    const cos = new Float64Array(size); const sin = new Float64Array(size);
    for (let x = 0; x < size; x++) { const a = -2 * Math.PI * fx * x / size; cos[x] = Math.cos(a); sin[x] = Math.sin(a); }
    for (let y = 0; y < size; y++) {
      let rr = 0, ii = 0; const offset = y * size;
      for (let x = 0; x < size; x++) { const value = pixels[offset + x]; rr += value * cos[x]; ii += value * sin[x]; }
      re[y] = rr; im[y] = ii;
    }
    rowProjections.set(fx, { re, im });
  }
  return carriers.map(([fy, fx]) => {
    const row = rowProjections.get(fx)!; let re = 0, im = 0;
    for (let y = 0; y < size; y++) {
      const angle = -2 * Math.PI * fy * y / size; const c = Math.cos(angle), s = Math.sin(angle);
      re += row.re[y] * c - row.im[y] * s; im += row.re[y] * s + row.im[y] * c;
    }
    return Math.atan2(im, re);
  });
}

function angularDistance(a: number, b: number): number {
  return Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
}
