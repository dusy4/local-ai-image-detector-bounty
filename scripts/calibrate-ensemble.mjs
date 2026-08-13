import { readFile } from "node:fs/promises";

const baseline = JSON.parse(await readFile("reports/openfake-baseline-200-rows.json", "utf8")).rows;
const safe = JSON.parse(await readFile("reports/safe-openfake-200.json", "utf8")).rows;
const ferret = JSON.parse(await readFile("reports/ferretnet-openfake-200.json", "utf8")).rows;
const safeByPath = new Map(safe.map(row => [row.file, row.score]));
const ferretByPath = new Map(ferret.map(row => [row.file, row.score]));
const rows = baseline.map(row => {
  const path = row.path.replace(/^benchmark\//, "").replace(/^ai\//, "fake/");
  return { truth: row.truth, row: row.row, baseline: row.score, safe: safeByPath.get(path), ferret: ferretByPath.get(path) };
});
if (rows.some(row => row.safe === undefined || row.ferret === undefined)) throw new Error("Benchmark rows do not align");
const calibration = rows.filter(row => row.row % 2 === 0);
const test = rows.filter(row => row.row % 2 !== 0);
const thresholdLogit = logit(0.65);
let best;
for (const baselineWeight of [0.5, 0.75, 1, 1.25, 1.5]) {
  for (let safeWeight = -2; safeWeight <= 2; safeWeight += 0.25) {
    for (let ferretWeight = -2; ferretWeight <= 2; ferretWeight += 0.25) {
      const values = calibration.map(row => baselineWeight * logit(row.baseline) + safeWeight * logit(row.safe) + ferretWeight * logit(row.ferret));
      for (const boundary of values) {
        const bias = thresholdLogit - boundary;
        const metrics = measure(calibration, baselineWeight, safeWeight, ferretWeight, bias);
        if (!best || metrics.balancedAccuracy > best.calibration.balancedAccuracy) {
          best = { baselineWeight, safeWeight, ferretWeight, bias, calibration: metrics };
        }
      }
    }
  }
}
best.test = measure(test, best.baselineWeight, best.safeWeight, best.ferretWeight, best.bias);
best.baselineTest = measure(test, 1, 0, 0, 0);
console.log(JSON.stringify(best, null, 2));

function measure(part, baselineWeight, safeWeight, ferretWeight, bias) {
  const positives = part.filter(row => row.truth === 1), negatives = part.filter(row => row.truth === 0);
  const predicts = row => bias + baselineWeight * logit(row.baseline) + safeWeight * logit(row.safe) + ferretWeight * logit(row.ferret) >= thresholdLogit;
  const tpr = positives.filter(predicts).length / positives.length;
  const tnr = negatives.filter(row => !predicts(row)).length / negatives.length;
  return { images: part.length, tpr, tnr, balancedAccuracy: (tpr + tnr) / 2 };
}

function logit(value) {
  const clamped = Math.max(1e-7, Math.min(1 - 1e-7, value));
  return Math.log(clamped / (1 - clamped));
}
