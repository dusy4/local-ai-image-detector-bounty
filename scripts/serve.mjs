import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, resolve } from "node:path";
const root = resolve(".");
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".jpg": "image/jpeg", ".wasm": "application/wasm", ".onnx": "application/octet-stream" };
createServer(async (request, response) => {
  try {
    const path = resolve(root, decodeURIComponent(new URL(request.url, "http://localhost").pathname.slice(1) || "dev/benchmark.html"));
    if (!path.startsWith(root)) throw new Error("outside root");
    const info = await stat(path); if (!info.isFile()) throw new Error("not file");
    response.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream", "cache-control": "no-store" });
    response.end(await readFile(path));
  } catch { response.writeHead(404); response.end("not found"); }
}).listen(4173, "127.0.0.1", () => console.log("http://127.0.0.1:4173/dev/benchmark.html"));
