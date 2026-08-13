import * as ort from "/public/vendor/ort.all.min.mjs";

const result = document.querySelector("#result");
try {
  ort.env.wasm.wasmPaths = "/public/wasm/";
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create("/public/models/community-forensics-official/model.onnx", { executionProviders: ["wasm"] });
  const output = await session.run({ pixel_values: new ort.Tensor("float32", new Float32Array(3 * 224 * 224), [1, 3, 224, 224]) });
  result.textContent = JSON.stringify({ logit: output.logit.data[0] });
  await session.release();
} catch (error) {
  result.textContent = `ERROR: ${error?.stack ?? error}`;
}
