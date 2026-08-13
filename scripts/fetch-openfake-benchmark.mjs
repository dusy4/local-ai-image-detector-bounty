import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const targetPerClass = Number(process.argv[2] ?? 50);
const outputRoot = process.argv[3] ?? "benchmark";
const dataset = "ComplexDataLab/OpenFake";
const config = "core";
const split = "test";
const base = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(dataset)}&config=${config}&split=${split}`;

const infoResponse = await fetch(`https://datasets-server.huggingface.co/info?dataset=${encodeURIComponent(dataset)}`);
if (!infoResponse.ok) throw new Error(`Dataset info ${infoResponse.status}`);
const info = await infoResponse.json();
const totalRows = info.dataset_info[config].splits[split].num_examples;
const pageSize = 100;
const pageCount = Math.ceil(totalRows / pageSize);

// Walk the complete split in a deterministic permutation so early-stop samples are
// distributed across the dataset instead of inheriting the dataset's row ordering.
const offsets = [];
const stride = 541; // coprime with the current page count (914)
for (let index = 0; index < pageCount; index++) offsets.push(((index * stride) % pageCount) * pageSize);

const candidates = { real: new Map(), fake: new Map() };
for (let cursor = 0; cursor < offsets.length && !hasEnoughCoverage(candidates, targetPerClass); cursor += 1) {
  const pages = await Promise.all(offsets.slice(cursor, cursor + 1).map(async offset => {
    const response = await fetchWithRetry(`${base}&offset=${offset}&length=${pageSize}`);
    if (!response.ok) throw new Error(`Dataset Viewer ${response.status} at offset ${offset}`);
    return response.json();
  }));
  for (const page of pages) {
    for (const item of page.rows) {
      const label = item.row.label;
      if (!(label in candidates)) continue;
      const source = item.row.model || "unknown";
      if (!candidates[label].has(source)) candidates[label].set(source, []);
      candidates[label].get(source).push({ row: item.row_idx, label, source, src: item.row.image.src });
    }
  }
  await new Promise(resolve => setTimeout(resolve, 750));
}

const selected = {
  real: roundRobin(candidates.real, targetPerClass),
  fake: roundRobin(candidates.fake, targetPerClass),
};
if (selected.real.length < targetPerClass || selected.fake.length < targetPerClass) {
  throw new Error(`Insufficient samples: real=${selected.real.length}, fake=${selected.fake.length}`);
}

const manifest = [];
for (const label of ["real", "fake"]) {
  await mkdir(join(outputRoot, label), { recursive: true });
  for (let cursor = 0; cursor < selected[label].length; cursor += 4) {
    const batch = selected[label].slice(cursor, cursor + 4);
    const rows = await Promise.all(batch.map(async (item, batchIndex) => {
      const response = await fetchWithRetry(item.src);
      if (!response.ok) throw new Error(`Image ${item.row}: ${response.status}`);
      const index = cursor + batchIndex;
      const file = `${String(index).padStart(4, "0")}-${item.row}.jpg`;
      await writeFile(join(outputRoot, label, file), new Uint8Array(await response.arrayBuffer()));
      return { path: `${outputRoot}/${label}/${file}`, truth: label === "fake" ? 1 : 0, source: item.source, row: item.row };
    }));
    manifest.push(...rows);
    console.log(`${label} ${Math.min(cursor + batch.length, targetPerClass)}/${targetPerClass}`);
  }
}

await writeFile(join(outputRoot, "manifest.json"), JSON.stringify({
  dataset,
  config,
  split,
  sampling: "deterministic split-wide page permutation with source round-robin",
  samples: manifest,
}, null, 2));

function roundRobin(groups, limit) {
  const queues = [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, rows]) => [...rows]);
  const result = [];
  while (result.length < limit && queues.some(queue => queue.length)) {
    for (const queue of queues) {
      if (result.length >= limit) break;
      const row = queue.shift();
      if (row) result.push(row);
    }
  }
  return result;
}

function hasEnoughCoverage(groups, target) {
  return ["real", "fake"].every(label => {
    const count = [...groups[label].values()].reduce((sum, rows) => sum + rows.length, 0);
    return count >= Math.ceil(target * 1.3) && groups[label].size >= 4;
  });
}

async function fetchWithRetry(url) {
  let response;
  for (let attempt = 0; attempt < 6; attempt++) {
    response = await fetch(url);
    if (response.status !== 429 && response.status < 500) return response;
    const retryAfter = Number(response.headers.get("retry-after"));
    await new Promise(resolve => setTimeout(resolve, Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 5000 * 2 ** attempt));
  }
  return response;
}
