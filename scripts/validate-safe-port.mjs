import { readFile } from "node:fs/promises";
import { safeDiagonalBand } from "../src/safe.ts";

const rgba = new Uint8Array(await readFile("research/safe-input.rgba"));
const expectedBytes = await readFile("research/safe-expected.f32");
const expected = new Float32Array(expectedBytes.buffer, expectedBytes.byteOffset, expectedBytes.byteLength / 4);
const actual = safeDiagonalBand(rgba);
let maximum = 0, mean = 0;
for (let index = 0; index < actual.length; index++) {
  const difference = Math.abs(actual[index] - expected[index]);
  maximum = Math.max(maximum, difference);
  mean += difference;
}
mean /= actual.length;
console.log(JSON.stringify({ maximum, mean }));
if (maximum > 1e-5) process.exitCode = 1;
