import { createHash } from "node:crypto";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { extname, relative, resolve, sep } from "node:path";

const root = resolve(process.argv[2] ?? "provider-input");
const output = resolve(process.argv[3] ?? `${root}/manifest.json`);
const extensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const benchmark = JSON.parse(await readFile("benchmark/manifest.json", "utf8"));
const benchmarkHashes = new Set();
for (const sample of benchmark.samples) benchmarkHashes.add(await digest(resolve(sample.path)));

const samples = [];
const seen = new Set();
for (const truth of [0, 1]) {
  const classRoot = resolve(root, truth ? "ai" : "real");
  for (const path of await walk(classRoot)) {
    const hash = await digest(path);
    if (seen.has(hash)) throw new Error(`Duplicate calibration image: ${path}`);
    if (benchmarkHashes.has(hash)) throw new Error(`Benchmark leakage detected: ${path}`);
    seen.add(hash);
    const local = relative(root, path).split(sep).join("/");
    const parts = local.split("/");
    samples.push({ path: relative(process.cwd(), path).split(sep).join("/"), truth, source: truth ? parts[1] ?? "unknown-ai" : parts[1] ?? "owned-real", sha256: hash });
  }
}
if (!samples.some(sample => sample.truth === 0) || !samples.some(sample => sample.truth === 1)) {
  throw new Error("Expected provider-input/real/<source> and provider-input/ai/<provider> images");
}
await writeFile(output, `${JSON.stringify({ dataset: "owned provider calibration", samples }, null, 2)}\n`);
console.log(JSON.stringify({ output, real: samples.filter(sample => !sample.truth).length, ai: samples.filter(sample => sample.truth).length }, null, 2));

async function walk(folder) {
  const files = [];
  for (const entry of await readdir(folder, { withFileTypes: true })) {
    const path = resolve(folder, entry.name);
    if (entry.isDirectory()) files.push(...await walk(path));
    else if (extensions.has(extname(entry.name).toLowerCase())) files.push(path);
  }
  return files.sort();
}

async function digest(path) {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}
