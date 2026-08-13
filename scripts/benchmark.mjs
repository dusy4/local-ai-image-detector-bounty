import { pipeline, env } from "@huggingface/transformers";
import { readdir } from "node:fs/promises";
import { join } from "node:path";

const root = process.argv[2] ?? "benchmark";
env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = "public/models/";
const detector = await pipeline("image-classification", "detector", { dtype: "q4", device: "wasm" });
const rows = [];
for (const [folder, truth] of [["real", 0], ["ai", 1]]) {
  for (const file of await readdir(join(root, folder))) {
    const output = await detector(join(root, folder, file), { topk: 2 });
    const fake = output.find(item => /fake|ai|artificial|generated/i.test(item.label));
    rows.push({ file: `${folder}/${file}`, truth, score: fake?.score ?? 1 - output[0].score });
  }
}
const threshold = 0.65;
const positives = rows.filter(r => r.truth), negatives = rows.filter(r => !r.truth);
const tpr = positives.filter(r => r.score >= threshold).length / positives.length;
const tnr = negatives.filter(r => r.score < threshold).length / negatives.length;
console.log(JSON.stringify({ threshold, images: rows.length, tpr, tnr, balancedAccuracy: (tpr + tnr) / 2 }, null, 2));
await detector.dispose();
