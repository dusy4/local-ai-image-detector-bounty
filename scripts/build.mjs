import { stripTypeScriptTypes } from "node:module";
import { cp, readFile, rm, writeFile } from "node:fs/promises";

const dist = new URL("../dist/", import.meta.url);
await rm(dist, { recursive: true, force: true });
await cp(new URL("../public/", import.meta.url), dist, { recursive: true });
for (const unused of ["models/community-forensics", "models/ferretnet", "models/safe"]) {
  await rm(new URL(`../dist/${unused}/`, import.meta.url), { recursive: true, force: true });
}
for (const file of ["service-worker", "content-script", "offscreen", "popup", "options", "shared", "metadata", "synthid", "ferretnet", "safe", "community-forensics", "modern-convnext", "sdxl-watermark", "c2pa"]) {
  const source = await readFile(new URL(`../src/${file}.ts`, import.meta.url), "utf8");
  const javascript = stripTypeScriptTypes(source, { mode: "transform", sourceMap: true, sourceUrl: `src/${file}.ts` });
  await writeFile(new URL(`../dist/${file}.js`, import.meta.url), javascript);
}
