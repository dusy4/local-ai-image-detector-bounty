import { env, pipeline } from "/public/vendor/transformers.min.js";
env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = "/public/models/"; env.backends.onnx.wasm.wasmPaths = "/public/wasm/";
const out = document.querySelector("#result");
try {
  const params = new URLSearchParams(location.search);
  const model = params.get("model") ?? "detector";
  const dtype = params.get("dtype") ?? "q4";
  const manifestUrl = params.get("manifest") ?? "/benchmark/manifest.json";
  const detector = await pipeline("image-classification", model, { dtype, device: "wasm" });
  const rawManifest = await fetch(manifestUrl).then(r => r.json());
  const limitPerClass = Number(params.get("limitPerClass") ?? 0);
  const classCounts = new Map();
  const samples = limitPerClass > 0 ? rawManifest.samples.filter(sample => {
    const count = classCounts.get(sample.truth) ?? 0;
    if (count >= limitPerClass) return false;
    classCounts.set(sample.truth, count + 1);
    return true;
  }) : rawManifest.samples;
  const manifest = { ...rawManifest, samples };
  const rows = [];
  const batchSize = 4;
  for (let index = 0; index < manifest.samples.length; index += batchSize) {
    const samples = manifest.samples.slice(index, index + batchSize);
    const batch = await detector(samples.map(sample => `/${sample.path}`), { topk: 2 });
    for (let i = 0; i < samples.length; i++) {
      const results = batch[i];
      const fake = results.find(item => /fake|ai|artificial|generated/i.test(item.label));
      const real = results.find(item => /real|human|authentic/i.test(item.label));
      rows.push({ ...samples[i], score: fake?.score ?? (real ? 1 - real.score : 0.5) });
    }
    out.textContent = `Processed ${Math.min(index + batchSize, manifest.samples.length)}/${manifest.samples.length}`;
  }
  const positives = rows.filter(r => r.truth), negatives = rows.filter(r => !r.truth), threshold = 0.65;
  const tpr = positives.filter(r => r.score >= threshold).length / positives.length;
  const tnr = negatives.filter(r => r.score < threshold).length / negatives.length;
  const bySource = Object.fromEntries([...new Set(rows.map(r => r.source))].map(source => { const part = rows.filter(r => r.source === source); return [source, { n: part.length, accuracy: part.filter(r => (r.score >= threshold ? 1 : 0) === r.truth).length / part.length }]; }));
  const report = { model, dtype, manifest: manifestUrl, threshold, images: rows.length, tpr, tnr, balancedAccuracy: (tpr + tnr) / 2, bySource, rows };
  window.__benchmarkReport = report;
  out.textContent = JSON.stringify(report, null, 2); document.title = `DONE ${(report.balancedAccuracy * 100).toFixed(1)}%`;
  await detector.dispose();
} catch (error) { out.textContent = error.stack ?? String(error); document.title = "ERROR"; }
