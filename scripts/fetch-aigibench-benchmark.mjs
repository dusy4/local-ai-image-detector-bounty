import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const targetPerClass = Number(process.argv[2] ?? 250);
const outputRoot = process.argv[3] ?? "benchmark-aigibench";
const endpoint = "https://datasets-server.huggingface.co/rows?dataset=TheKernel01%2FAIGIBench&config=default&split=validation";
const totalRows = 20_000;
const pageSize = 100;
const offsets = [];
for (let index = 0; index < totalRows / pageSize; index++) offsets.push(((index * 73) % 200) * pageSize);

const selected = { real: [], fake: [] };
for (let cursor = 0; cursor < offsets.length && (selected.real.length < targetPerClass || selected.fake.length < targetPerClass); cursor += 4) {
  const pages = await Promise.all(offsets.slice(cursor, cursor + 4).map(async offset => {
    const response = await fetchWithRetry(`${endpoint}&offset=${offset}&length=${pageSize}`);
    if (!response.ok) throw new Error(`Dataset Viewer ${response.status} at offset ${offset}`);
    return response.json();
  }));
  for (const page of pages) {
    for (const item of page.rows) {
      const label = item.row.label === 1 ? "fake" : "real";
      if (selected[label].length >= targetPerClass) continue;
      const generators = ["real", "progan", "sd14"];
      selected[label].push({ row: item.row_idx, source: generators[item.row.generator] ?? `generator-${item.row.generator}`, src: item.row.image.src });
    }
  }
}

const manifest = [];
for (const label of ["real", "fake"]) {
  await mkdir(join(outputRoot, label), { recursive: true });
  for (let cursor = 0; cursor < selected[label].length; cursor += 8) {
    const batch = selected[label].slice(cursor, cursor + 8);
    manifest.push(...await Promise.all(batch.map(async (item, batchIndex) => {
      const response = await fetchWithRetry(item.src);
      if (!response.ok) throw new Error(`Image ${item.row}: ${response.status}`);
      const index = cursor + batchIndex;
      const file = `${String(index).padStart(4, "0")}-${item.row}.jpg`;
      await writeFile(join(outputRoot, label, file), new Uint8Array(await response.arrayBuffer()));
      return { path: `${outputRoot}/${label}/${file}`, truth: label === "fake" ? 1 : 0, source: item.source, row: item.row };
    })));
    console.log(`${label} ${Math.min(cursor + batch.length, targetPerClass)}/${targetPerClass}`);
  }
}

await writeFile(join(outputRoot, "manifest.json"), JSON.stringify({
  dataset: "TheKernel01/AIGIBench",
  config: "default",
  split: "validation",
  sampling: "deterministic split-wide page permutation",
  samples: manifest,
}, null, 2));

async function fetchWithRetry(url) {
  let response;
  for (let attempt = 0; attempt < 5; attempt++) {
    response = await fetch(url);
    if (response.status !== 429 && response.status < 500) return response;
    await new Promise(resolve => setTimeout(resolve, 1000 * 2 ** attempt));
  }
  return response;
}
