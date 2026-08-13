import type { Signal } from "./shared.js";

const DIRECT_MARKERS: Array<[RegExp, string, string, number]> = [
  [/trainedAlgorithmicMedia/i, "C2PA", "unverified trained-algorithmic-media assertion", 0.5],
  [/\b(?:DALL[ -]?E|Midjourney|Stable Diffusion|ComfyUI|Automatic1111|InvokeAI)\b/i, "metadata", "generator metadata", 0.98],
  [/\b(?:Adobe Firefly|Google AI|Gemini|Imagen|OpenAI)\b/i, "metadata", "provider provenance metadata", 0.94],
  [/parameters[\s\S]{0,400}(?:steps|sampler|cfg scale|seed)/i, "metadata", "diffusion generation parameters", 0.99]
];

export function inspectMetadata(bytes: Uint8Array): Signal[] {
  const scan = latin1(bytes.subarray(0, Math.min(bytes.length, 12 * 1024 * 1024)));
  const signals: Signal[] = [];
  for (const [pattern, source, detail, score] of DIRECT_MARKERS) {
    if (pattern.test(scan)) signals.push({ source, score, detail });
  }
  if (/c2pa|content credentials|contentauth/i.test(scan) && signals.length === 0) {
    signals.push({ source: "C2PA", score: 0.2, detail: "Content Credentials present, but no AI assertion found" });
  }
  return dedupe(signals);
}

function latin1(bytes: Uint8Array): string {
  let out = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    out += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return out;
}

function dedupe(signals: Signal[]): Signal[] {
  return signals.filter((signal, index) => signals.findIndex(other => other.detail === signal.detail) === index);
}
