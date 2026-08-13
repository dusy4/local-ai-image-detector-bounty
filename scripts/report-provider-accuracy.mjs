import { readFile, writeFile } from "node:fs/promises";
import { basename } from "node:path";

const baseline = JSON.parse(await readFile("reports/openfake-baseline-200-rows.json", "utf8")).rows;
const community = byFile(JSON.parse(await readFile("reports/community-forensics-openfake-200.json", "utf8")).rows);
const modern = byFile(JSON.parse(await readFile("reports/modern-convnext-linear-int8-openfake-200.json", "utf8")).rows);
const rows = baseline.map(row => {
  const file = basename(row.path);
  const communityScore = community.get(file).score, modernScore = modern.get(file).score;
  const score = sigmoid(0.36182409415728856 + 1.2 * logit(row.score) + 0.2 * logit(communityScore) + 0.2 * logit(modernScore));
  return { source: row.source, truth: row.truth, score };
});
const bySource = Object.fromEntries([...new Set(rows.map(row => row.source))].sort().map(source => {
  const part = rows.filter(row => row.source === source);
  return [source, { n: part.length, truth: part[0].truth ? "AI" : "real", accuracy: part.filter(row => (row.score >= 0.65 ? 1 : 0) === row.truth).length / part.length }];
}));
await writeFile("reports/provider-accuracy.json", JSON.stringify({ threshold: 0.65, bySource }, null, 2));
console.log(JSON.stringify(bySource, null, 2));

function byFile(rows) { return new Map(rows.map(row => [basename(row.file), row])); }
function logit(value) { const bounded = Math.max(1e-7, Math.min(1 - 1e-7, value)); return Math.log(bounded / (1 - bounded)); }
function sigmoid(value) { return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value)); }
