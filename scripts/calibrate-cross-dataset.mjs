import { readFile, writeFile } from "node:fs/promises";
import { basename } from "node:path";

const datasets = [
  loadDataset("openfake", "reports/openfake-baseline-200-rows.json", "reports/community-forensics-openfake-200.json"),
  loadDataset("aigibench", "reports/aigibench-baseline-300-rows.json", "reports/aigibench-community-forensics-300.json"),
];
const resolved = await Promise.all(datasets);
const calibration = resolved.flatMap(dataset => dataset.rows.filter(row => row.splitIndex % 2 === 0));
const testByDataset = Object.fromEntries(resolved.map(dataset => [dataset.name, dataset.rows.filter(row => row.splitIndex % 2 !== 0)]));
const targetLogit = logit(0.65);

let best;
for (let baselineWeight = 0; baselineWeight <= 2.0001; baselineWeight += 0.1) {
  for (let communityWeight = 0; communityWeight <= 2.0001; communityWeight += 0.1) {
    if (baselineWeight + communityWeight === 0) continue;
    const boundaries = calibration.map(row => targetLogit - baselineWeight * logit(row.baseline) - communityWeight * logit(row.community));
    for (const bias of boundaries) {
      const perDataset = Object.fromEntries(resolved.map(dataset => {
        const part = calibration.filter(row => row.dataset === dataset.name);
        return [dataset.name, measure(part, baselineWeight, communityWeight, bias)];
      }));
      const macro = average(Object.values(perDataset).map(metrics => metrics.balancedAccuracy));
      const floor = Math.min(...Object.values(perDataset).map(metrics => metrics.balancedAccuracy));
      if (!best || macro > best.macro || (macro === best.macro && floor > best.floor)) {
        best = { baselineWeight, communityWeight, bias, macro, floor, perDataset };
      }
    }
  }
}

best.test = Object.fromEntries(Object.entries(testByDataset).map(([name, rows]) => [name, measure(rows, best.baselineWeight, best.communityWeight, best.bias)]));
best.testMacro = average(Object.values(best.test).map(metrics => metrics.balancedAccuracy));
best.baselines = Object.fromEntries(Object.entries(testByDataset).map(([name, rows]) => [name, {
  active: measure(rows, 1, 0, 0),
  community: measure(rows, 0, 1, 0),
  monotonicOr: measureOr(rows),
}]));
best.fullDatasetOr = Object.fromEntries(resolved.map(dataset => [dataset.name, measureOr(dataset.rows)]));

await writeFile("reports/cross-dataset-calibration.json", JSON.stringify(best, null, 2));
console.log(JSON.stringify(best, null, 2));

async function loadDataset(name, baselinePath, communityPath) {
  const baseline = JSON.parse(await readFile(baselinePath, "utf8")).rows;
  const community = JSON.parse(await readFile(communityPath, "utf8")).rows;
  const communityByFile = new Map(community.map(row => [basename(row.file), row]));
  const splitCounts = new Map();
  const rows = baseline.map(row => {
    const peer = communityByFile.get(basename(row.path));
    if (!peer) throw new Error(`Missing Community Forensics score for ${row.path}`);
    const splitKey = `${row.truth}|${row.source}`;
    const splitIndex = splitCounts.get(splitKey) ?? 0;
    splitCounts.set(splitKey, splitIndex + 1);
    return { dataset: name, row: row.row, truth: row.truth, source: row.source, splitIndex, baseline: row.score, community: peer.score };
  });
  return { name, rows };
}

function measure(rows, baselineWeight, communityWeight, bias) {
  const positives = rows.filter(row => row.truth === 1);
  const negatives = rows.filter(row => row.truth === 0);
  const predicted = row => bias + baselineWeight * logit(row.baseline) + communityWeight * logit(row.community) >= targetLogit;
  const tpr = positives.filter(predicted).length / positives.length;
  const tnr = negatives.filter(row => !predicted(row)).length / negatives.length;
  return { images: rows.length, tpr, tnr, balancedAccuracy: (tpr + tnr) / 2 };
}

function measureOr(rows) {
  const positives = rows.filter(row => row.truth === 1);
  const negatives = rows.filter(row => row.truth === 0);
  const predicted = row => row.baseline >= 0.65 || row.community >= 0.65;
  const tpr = positives.filter(predicted).length / positives.length;
  const tnr = negatives.filter(row => !predicted(row)).length / negatives.length;
  return { images: rows.length, tpr, tnr, balancedAccuracy: (tpr + tnr) / 2 };
}

function logit(value) {
  const clamped = Math.max(1e-7, Math.min(1 - 1e-7, value));
  return Math.log(clamped / (1 - clamped));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
