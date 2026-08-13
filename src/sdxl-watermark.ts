import type { Signal } from "./shared.js";

const MESSAGE = "101100111110110010010000011110111011000110011110";
const SCALE = 36;

export function detectSdxlWatermark(bitmap: ImageBitmap): Signal | undefined {
  if (bitmap.width * bitmap.height < 256 * 256) return;
  const width = Math.floor(bitmap.width / 8) * 8;
  const height = Math.floor(bitmap.height / 8) * 8;
  if (!width || !height) return;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return;
  context.drawImage(bitmap, 0, 0);
  const rgba = context.getImageData(0, 0, width, height).data;
  const matches = decodeSdxlMatch(rgba, width, height);
  if (matches <= 33) return;
  const score = matches >= 36 ? 0.999999 : 0.9998;
  return {
    source: "SDXL watermark",
    score,
    detail: `${matches}/48 fixed Stability AI DWT/DCT bits matched`,
  };
}

export function decodeSdxlMatch(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): number {
  const decoded = decodeSdxlBits(rgba, width, height);
  let matches = 0;
  for (let index = 0; index < MESSAGE.length; index++) {
    if ((decoded[index] ? "1" : "0") === MESSAGE[index]) matches++;
  }
  return matches;
}

export function decodeSdxlBits(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): boolean[] {
  const ones = new Uint32Array(MESSAGE.length);
  const totals = new Uint32Array(MESSAGE.length);
  let sequence = 0;

  for (let top = 0; top < height; top += 8) {
    for (let left = 0; left < width; left += 8) {
      const block = new Float64Array(16);
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
          const u00 = chromaU(rgba, width, left + x * 2, top + y * 2);
          const u01 = chromaU(rgba, width, left + x * 2 + 1, top + y * 2);
          const u10 = chromaU(rgba, width, left + x * 2, top + y * 2 + 1);
          const u11 = chromaU(rgba, width, left + x * 2 + 1, top + y * 2 + 1);
          const upper = (u00 + u01) * Math.SQRT1_2;
          const lower = (u10 + u11) * Math.SQRT1_2;
          block[y * 4 + x] = (upper + lower) * Math.SQRT1_2;
        }
      }
      const bit = inferBit(block);
      const index = sequence++ % MESSAGE.length;
      ones[index] += bit;
      totals[index]++;
    }
  }

  const decoded = [];
  for (let index = 0; index < MESSAGE.length; index++) {
    decoded.push(ones[index] * 255 > totals[index] * 127);
  }
  return decoded;
}

function chromaU(rgba: Uint8ClampedArray | Uint8Array, width: number, x: number, y: number): number {
  const pixel = (y * width + x) * 4;
  const red = rgba[pixel], green = rgba[pixel + 1], blue = rgba[pixel + 2];
  const luminance = (red * 4899 + green * 9617 + blue * 1868 + 8192) >> 14;
  return clampByte(((blue - luminance) * 8061 + (128 << 14) + 8192) >> 14);
}

function inferBit(block: Float64Array): number {
  let selected = block[1];
  for (let index = 2; index < block.length; index++) {
    if (Math.abs(block[index]) > Math.abs(selected)) {
      selected = block[index];
    }
  }
  return Math.abs(selected) % SCALE > SCALE / 2 ? 1 : 0;
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, value));
}
