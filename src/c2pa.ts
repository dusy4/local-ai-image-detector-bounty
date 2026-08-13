import type { Signal } from "./shared.js";

let sdkPromise: Promise<any> | undefined;

function getSdk() {
  if (!sdkPromise) {
    sdkPromise = import(chrome.runtime.getURL("vendor/c2pa/index.js"))
      .then((module: any) => module.createC2pa({
        wasmSrc: chrome.runtime.getURL("vendor/c2pa/c2pa_bg.wasm"),
        workerSrc: new URL(chrome.runtime.getURL("vendor/c2pa/c2pa_worker.js")),
      }));
  }
  return sdkPromise;
}

export async function inspectVerifiedC2pa(blob: Blob): Promise<Signal[]> {
  const sdk = await getSdk();
  const reader = await sdk.reader.fromBlob(blob.type, blob);
  if (!reader) return [];
  try {
    const store = await reader.manifestStore();
    const state = String(store.validation_state ?? "unknown").toLowerCase();
    const failures = store.validation_results?.activeManifest?.failure ?? [];
    const trusted = failures.length === 0 && state === "trusted";
    const active = store.active_manifest && store.manifests?.[store.active_manifest];
    const serialized = JSON.stringify(active ?? store);
    const synthetic = /trainedAlgorithmicMedia|trainedAlgorithmicData|algorithmicMedia|compositeSynthetic/i.test(serialized);
    if (trusted && synthetic) {
      return [{ source: "C2PA", score: 0.999999, detail: `validated ${state} manifest declares synthetic media` }];
    }
    return [{ source: "C2PA", score: synthetic ? 0.5 : 0.1, detail: `${state} manifest${synthetic ? " contains a synthetic claim without trusted signer status" : " found without an AI-origin claim"}` }];
  } finally {
    await reader.free();
  }
}
