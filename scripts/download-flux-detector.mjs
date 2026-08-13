import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";

const repo = "LukasT9/flux-detector";
const revision = "f3baa7a75757130338554a97f8a9fa4416dc63d9";
const response = await fetch(`https://huggingface.co/${repo}/resolve/${revision}/model.pt?download=true`);
if (!response.ok) throw new Error(`Download failed: ${response.status}`);
const bytes = new Uint8Array(await response.arrayBuffer());
await mkdir(new URL("../research/flux-detector/", import.meta.url), { recursive: true });
await writeFile(new URL("../research/flux-detector/model.pt", import.meta.url), bytes);
const lock = { repo, revision, file: "model.pt", bytes: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
await writeFile(new URL("../reports/flux-detector-lock.json", import.meta.url), JSON.stringify(lock, null, 2) + "\n");
console.log(JSON.stringify(lock));
