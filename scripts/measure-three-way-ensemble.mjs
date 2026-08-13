import { readFile, writeFile } from "node:fs/promises";
import { basename } from "node:path";

const specs = [
  ["openfake", "reports/openfake-baseline-200-rows.json", "reports/community-forensics-openfake-200.json", "reports/modern-convnext-openfake-200.json"],
  ["aigibench", "reports/aigibench-baseline-300-rows.json", "reports/aigibench-community-forensics-300.json", "reports/modern-convnext-aigibench-300.json"],
];
const report = {};
for (const [name, baselinePath, communityPath, modernPath] of specs) {
  const baseline = JSON.parse(await readFile(baselinePath, "utf8")).rows;
  const community = byFile(JSON.parse(await readFile(communityPath, "utf8")).rows);
  const modern = byFile(JSON.parse(await readFile(modernPath, "utf8")).rows);
  const rows = baseline.map(row => ({
    truth: row.truth,
    baseline: row.score,
    community: community.get(basename(row.path)).score,
    modern: modern.get(basename(row.path)).score,
  }));
  report[name] = {
    baselineCommunity: measure(rows, ["baseline", "community"]),
    baselineModern: measure(rows, ["baseline", "modern"]),
    communityModern: measure(rows, ["community", "modern"]),
    all: measure(rows, ["baseline", "community", "modern"]),
  };
}
await writeFile("reports/three-way-ensemble.json", JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));

function byFile(rows) { return new Map(rows.map(row => [basename(row.file), row])); }
function measure(rows, experts) {
  const positives = rows.filter(row => row.truth), negatives = rows.filter(row => !row.truth);
  const predict = row => experts.some(expert => row[expert] >= 0.65);
  const tpr = positives.filter(predict).length / positives.length;
  const tnr = negatives.filter(row => !predict(row)).length / negatives.length;
  return { tpr, tnr, balancedAccuracy: (tpr + tnr) / 2 };
}
