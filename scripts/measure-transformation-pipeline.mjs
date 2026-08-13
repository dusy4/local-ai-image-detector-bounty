import { readFile, writeFile } from "node:fs/promises";
import { basename } from "node:path";

const baseline = JSON.parse(await readFile("reports/transforms-baseline-250.json", "utf8")).rows;
const community = byFile(JSON.parse(await readFile("reports/transforms-community-250.json", "utf8")).rows);
const modern = byFile(JSON.parse(await readFile("reports/transforms-modern-250.json", "utf8")).rows);
const manifest = JSON.parse(await readFile("benchmark-transforms/manifest.json", "utf8")).samples;
const metadata = new Map(manifest.map(row => [basename(row.path), row]));
const rows = baseline.map(row => {
  const file = basename(row.path);
  const peerCommunity = community.get(file), peerModern = modern.get(file), meta = metadata.get(file);
  if (!peerCommunity || !peerModern || !meta) throw new Error(`Missing aligned row ${file}`);
  const score = sigmoid(0.36182409415728856 + 1.2 * logit(row.score) + 0.2 * logit(peerCommunity.score) + 0.2 * logit(peerModern.score));
  return { ...meta, baseline: row.score, community: peerCommunity.score, modern: peerModern.score, score };
});
const report = { overall: measure(rows), byTransform: Object.fromEntries([...new Set(rows.map(row => row.transform))].map(transform => [transform, measure(rows.filter(row => row.transform === transform))])), rows };
await writeFile("reports/transformation-pipeline-250.json", JSON.stringify(report, null, 2));
console.log(JSON.stringify({ overall: report.overall, byTransform: report.byTransform }, null, 2));

function byFile(rows) { return new Map(rows.map(row => [basename(row.file), row])); }
function measure(part) {
  const positives = part.filter(row => row.truth), negatives = part.filter(row => !row.truth);
  const tpr = positives.filter(row => row.score >= 0.65).length / positives.length;
  const tnr = negatives.filter(row => row.score < 0.65).length / negatives.length;
  return { images: part.length, tpr, tnr, balancedAccuracy: (tpr + tnr) / 2 };
}
function logit(value) { const bounded = Math.max(1e-7, Math.min(1 - 1e-7, value)); return Math.log(bounded / (1 - bounded)); }
function sigmoid(value) { return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value)); }
