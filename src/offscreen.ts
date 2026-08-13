import { inspectMetadata } from "./metadata.js";
import { detectSynthId } from "./synthid.js";
import { detectSdxlWatermark } from "./sdxl-watermark.js";
import { inspectVerifiedC2pa } from "./c2pa.js";
import { detectCommunityForensics } from "./community-forensics.js";
import { detectModernConvNext } from "./modern-convnext.js";
import { fuseScores, REQUIRED_THRESHOLD, type AnalysisResult } from "./shared.js";

const transformers = await import(chrome.runtime.getURL("vendor/transformers.min.js"));
const { env, pipeline, RawImage } = transformers;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = chrome.runtime.getURL("models/");
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL("wasm/");

let detectorPromise: Promise<any> | undefined;
const getDetector = () => detectorPromise ??= pipeline("image-classification", "detector", { dtype: "q4", device: "wasm" });

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "offscreen") return;
  if (message.type === "health") { sendResponse({ ready: true }); return; }
  if (message.type === "analyze") {
    analyze(message.url).then(sendResponse).catch(error => sendResponse({ error: String(error?.message ?? error) }));
    return true;
  }
});

async function analyze(url: string): Promise<AnalysisResult> {
  const started = performance.now();
  const response = await fetch(url, { credentials: "omit", cache: "force-cache" });
  if (!response.ok) throw new Error(`Image fetch failed (${response.status})`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const blob = new Blob([bytes], { type: response.headers.get("content-type") ?? "image/jpeg" });
  const bitmap = await createImageBitmap(blob);
  try {
    const detector = await getDetector();
    const image = await RawImage.fromBlob(blob);
    const outputs = await detector(image, { topk: 2 });
    const rows = Array.isArray(outputs[0]) ? outputs[0] : outputs;
    const fake = rows.find(row => /fake|ai|artificial|generated/i.test(row.label));
    const real = rows.find(row => /real|human|authentic/i.test(row.label));
    const modelScore = fake?.score ?? (real ? 1 - real.score : rows[0]?.score ?? 0.5);
    const experts = [];
    try { experts.push(await withTimeout(detectCommunityForensics(bitmap), 30_000, "Community Forensics timed out")); }
    catch (error) { experts.push({ source: "Community Forensics", score: 0.5, active: false, detail: `unavailable: ${String(error)}` }); }
    try { experts.push(await withTimeout(detectModernConvNext(bitmap), 60_000, "Modern ConvNeXt timed out")); }
    catch (error) { experts.push({ source: "Modern ConvNeXt", score: 0.5, active: false, detail: `unavailable: ${String(error)}` }); }
    const preliminaryScore = fuseScores(modelScore, [], experts);
    if (preliminaryScore >= 0.60 && preliminaryScore < REQUIRED_THRESHOLD) {
      try { experts[0] = await withTimeout(detectCommunityForensics(bitmap, true), 60_000, "Community Forensics TTA timed out"); }
      catch { /* retain the validated center-crop score */ }
    }
    const signals = inspectMetadata(bytes);
    if (signals.some(signal => signal.source === "C2PA")) {
      try { signals.push(...await inspectVerifiedC2pa(blob)); } catch { /* malformed or unsupported provenance */ }
    }
    const synthId = await detectSynthId(bitmap);
    if (synthId) signals.push(synthId);
    const sdxl = detectSdxlWatermark(bitmap);
    if (sdxl) signals.push(sdxl);
    const score = fuseScores(modelScore, signals, experts);
    return { score, modelScore, experts, signals, label: score >= REQUIRED_THRESHOLD ? "AI" : "real", elapsedMs: performance.now() - started };
  } finally { bitmap.close(); }
}

async function withTimeout<T>(promise: Promise<T>, milliseconds: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error(message)), milliseconds); }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
