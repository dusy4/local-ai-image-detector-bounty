import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = "onnx-community/Deep-Fake-Detector-v2-Model-ONNX";
const REVISION = "4ea3d66";
const FILES = ["config.json", "preprocessor_config.json", "onnx/model_q4.onnx"];
const root = new URL("../public/models/detector/", import.meta.url);
const lock = { repo: REPO, revision: REVISION, files: {} };

for (const file of FILES) {
  const response = await fetch(`https://huggingface.co/${REPO}/resolve/${REVISION}/${file}?download=true`, { redirect: "follow" });
  if (!response.ok) throw new Error(`${response.status} downloading ${file}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const target = join(fileURLToPath(root), file);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes);
  lock.files[file] = { bytes: bytes.byteLength, sha256: createHash("sha256").update(bytes).digest("hex") };
  console.log(`${file}: ${bytes.byteLength} bytes`);
}
await writeFile(new URL("../model-lock.json", import.meta.url), JSON.stringify(lock, null, 2) + "\n");
