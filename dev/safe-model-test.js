import * as ort from "/public/vendor/ort.all.min.mjs";

const result = document.querySelector("#result");
try {
  ort.env.wasm.wasmPaths = "/public/wasm/";
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create("/public/models/safe/model.onnx", { executionProviders: ["wasm"] });
  const output = await session.run({ diagonal_band: new ort.Tensor("float32", new Float32Array(3 * 256 * 256), [1, 3, 256, 256]) });
  result.textContent = JSON.stringify({ logits: Array.from(output.logits.data) });
  await session.release();
} catch (error) {
  result.textContent = `ERROR: ${error?.stack ?? error}`;
}
