import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const perGenerator = Number(process.argv[2] ?? 5);
const outputRoot = process.argv[3] ?? "benchmark-corebench";
const endpoint = "https://datasets-server.huggingface.co/rows?dataset=lioooox%2FT2I-CoReBench-Images&config=default&split=validation";
const groups = new Map();

for (let offset = 0; offset < 3700; offset += 100) {
  const response = await fetchWithRetry(`${endpoint}&offset=${offset}&length=100`);
  if (!response.ok) throw new Error(`Dataset Viewer ${response.status} at offset ${offset}`);
  const page = await response.json();
  for (const item of page.rows) {
    const key = item.row.__key__;
    const generator = key.split("/")[0];
    if (!groups.has(generator)) groups.set(generator, []);
    if (groups.get(generator).length < perGenerator) {
      groups.get(generator).push({ row: item.row_idx, source: generator, src: item.row.png.src });
    }
  }
}

const selected = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).flatMap(([, rows]) => rows);
await mkdir(join(outputRoot, "fake"), { recursive: true });
const manifest = [];
for (let cursor = 0; cursor < selected.length; cursor += 6) {
  const batch = selected.slice(cursor, cursor + 6);
  manifest.push(...await Promise.all(batch.map(async (item, batchIndex) => {
    const response = await fetchWithRetry(item.src);
    if (!response.ok) throw new Error(`Image ${item.row}: ${response.status}`);
    const index = cursor + batchIndex;
    const file = `${String(index).padStart(4, "0")}-${item.row}.jpg`;
    await writeFile(join(outputRoot, "fake", file), new Uint8Array(await response.arrayBuffer()));
    return { path: `${outputRoot}/fake/${file}`, truth: 1, source: item.source, row: item.row };
  })));
  console.log(`fake ${Math.min(cursor + batch.length, selected.length)}/${selected.length}`);
}
await writeFile(join(outputRoot, "manifest.json"), JSON.stringify({
  dataset: "lioooox/T2I-CoReBench-Images",
  config: "default",
  split: "validation",
  sampling: `first ${perGenerator} per generator after complete metadata scan`,
  samples: manifest,
}, null, 2));
console.log(`${groups.size} generators, ${selected.length} images`);

async function fetchWithRetry(url) {
  let response;
  for (let attempt = 0; attempt < 6; attempt++) {
    response = await fetch(url);
    if (response.status !== 429 && response.status < 500) return response;
    await new Promise(resolve => setTimeout(resolve, 1500 * 2 ** attempt));
  }
  return response;
}
