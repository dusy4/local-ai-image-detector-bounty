import { readFile, writeFile } from "node:fs/promises";
import { basename } from "node:path";

const specs = [
  ["openfake", "reports/openfake-baseline-200-rows.json", "reports/community-forensics-openfake-200.json", "reports/modern-convnext-linear-int8-openfake-200.json"],
  ["aigibench", "reports/aigibench-baseline-300-rows.json", "reports/aigibench-community-forensics-300.json", "reports/modern-convnext-linear-int8-aigibench-300.json"],
];
const datasets = [];
for (const [name, baselinePath, communityPath, modernPath] of specs) {
  const baseline = JSON.parse(await readFile(baselinePath, "utf8")).rows;
  const community = byFile(JSON.parse(await readFile(communityPath, "utf8")).rows);
  const modern = byFile(JSON.parse(await readFile(modernPath, "utf8")).rows);
  const counters = new Map();
  const rows = baseline.map(row => {
    const key = `${row.truth}|${row.source}`;
    const splitIndex = counters.get(key) ?? 0;
    counters.set(key, splitIndex + 1);
    return { dataset: name, truth: row.truth, splitIndex, baseline: row.score, community: community.get(basename(row.path)).score, modern: modern.get(basename(row.path)).score };
  });
  datasets.push({ name, rows });
}

const calibration = datasets.flatMap(dataset => dataset.rows.filter(row => row.splitIndex % 2 === 0));
const target = logit(0.65);
let best;
for (let baselineWeight = 0; baselineWeight <= 2.001; baselineWeight += 0.2) {
  for (let communityWeight = 0; communityWeight <= 2.001; communityWeight += 0.2) {
    for (let modernWeight = 0; modernWeight <= 2.001; modernWeight += 0.2) {
      if (baselineWeight + communityWeight + modernWeight === 0) continue;
      const sortedBiases = calibration.map(row => target - scoreWithoutBias(row, baselineWeight, communityWeight, modernWeight)).sort((a, b) => a - b);
      for (let index = 0; index < sortedBiases.length; index += 4) {
        const bias = sortedBiases[index];
        const perDataset = Object.fromEntries(datasets.map(dataset => [dataset.name, measure(calibration.filter(row => row.dataset === dataset.name), baselineWeight, communityWeight, modernWeight, bias)]));
        const values = Object.values(perDataset).map(metrics => metrics.balancedAccuracy);
        const floor = Math.min(...values), macro = average(values);
        if (!best || floor > best.floor || (floor === best.floor && macro > best.macro)) best = { baselineWeight, communityWeight, modernWeight, bias, floor, macro, calibration: perDataset };
      }
    }
  }
}
best.test = Object.fromEntries(datasets.map(dataset => [dataset.name, measure(dataset.rows.filter(row => row.splitIndex % 2 !== 0), best.baselineWeight, best.communityWeight, best.modernWeight, best.bias)]));
best.testFloor = Math.min(...Object.values(best.test).map(metrics => metrics.balancedAccuracy));
best.testMacro = average(Object.values(best.test).map(metrics => metrics.balancedAccuracy));
await writeFile("reports/three-way-calibration.json", JSON.stringify(best, null, 2));
console.log(JSON.stringify(best, null, 2));

function byFile(rows) { return new Map(rows.map(row => [basename(row.file), row])); }
function scoreWithoutBias(row, baselineWeight, communityWeight, modernWeight) { return baselineWeight * logit(row.baseline) + communityWeight * logit(row.community) + modernWeight * logit(row.modern); }
function measure(rows, baselineWeight, communityWeight, modernWeight, bias) {
  const positives = rows.filter(row => row.truth), negatives = rows.filter(row => !row.truth);
  const predict = row => bias + scoreWithoutBias(row, baselineWeight, communityWeight, modernWeight) >= target;
  const tpr = positives.filter(predict).length / positives.length;
  const tnr = negatives.filter(row => !predict(row)).length / negatives.length;
  return { images: rows.length, tpr, tnr, balancedAccuracy: (tpr + tnr) / 2 };
}
function logit(value) { const bounded = Math.max(1e-7, Math.min(1 - 1e-7, value)); return Math.log(bounded / (1 - bounded)); }
function average(values) { return values.reduce((sum, value) => sum + value, 0) / values.length; }
