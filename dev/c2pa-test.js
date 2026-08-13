import { createC2pa } from "/public/vendor/c2pa/index.js";

const result = document.querySelector("#result");
try {
  const sdk = await createC2pa({ wasmSrc: "/public/vendor/c2pa/c2pa_bg.wasm" });
  const blob = await fetch("/research/c2pa-test.jpg").then(response => response.blob());
  const reader = await sdk.reader.fromBlob(blob.type, blob);
  if (!reader) throw new Error("No manifest found");
  const store = await reader.manifestStore();
  result.textContent = JSON.stringify({ state: store.validation_state, active: store.active_manifest, manifests: Object.keys(store.manifests ?? {}).length });
  await reader.free();
} catch (error) {
  result.textContent = `ERROR: ${error?.stack ?? error}`;
}
