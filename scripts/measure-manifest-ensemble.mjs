import { basename } from "node:path";
import { readFile, writeFile } from "node:fs/promises";

const [baselinePath, communityPath, modernPath, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error("Usage: node scripts/measure-manifest-ensemble.mjs <baseline> <community> <modern> <output>");
const baseline = JSON.parse(await readFile(baselinePath, "utf8"));
const community = byName(JSON.parse(await readFile(communityPath, "utf8")).rows);
const modern = byName(JSON.parse(await readFile(modernPath, "utf8")).rows);
const rows = baseline.rows.map(row => {
  const file = basename(row.path);
  const expertCommunity = community.get(file)?.score;
  const expertModern = modern.get(file)?.score;
  if (expertCommunity == null || expertModern == null) throw new Error(`Missing expert score for ${file}`);
  const score = sigmoid(0.36182409415728856 + 1.2 * logit(row.score) + 0.2 * logit(expertCommunity) + 0.2 * logit(expertModern));
  return { ...row, baseline: row.score, community: expertCommunity, modern: expertModern, score, predicted: score >= 0.65 ? 1 : 0 };
});
const positives = rows.filter(row => row.truth), negatives = rows.filter(row => !row.truth);
const report = {
  threshold: 0.65,
  images: rows.length,
  tpr: positives.length ? positives.filter(row => row.predicted).length / positives.length : null,
  tnr: negatives.length ? negatives.filter(row => !row.predicted).length / negatives.length : null,
  rows,
};
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ images: report.images, tpr: report.tpr, tnr: report.tnr }, null, 2));

function byName(rows) { return new Map(rows.map(row => [basename(row.file ?? row.path), row])); }
function logit(value) { const safe = Math.max(1e-6, Math.min(1 - 1e-6, value)); return Math.log(safe / (1 - safe)); }
function sigmoid(value) { return 1 / (1 + Math.exp(-value)); }
