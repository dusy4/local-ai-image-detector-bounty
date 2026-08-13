import { readFile } from "node:fs/promises";
import { decodeSdxlBits, decodeSdxlMatch } from "../src/sdxl-watermark.ts";

const [path, width, height, expected] = process.argv.slice(2);
const bytes = new Uint8Array(await readFile(path));
const actual = decodeSdxlMatch(bytes, Number(width), Number(height));
const equivalent = Math.abs(actual - Number(expected)) <= 1;
console.log(JSON.stringify({ actual, expected: Number(expected), equivalent, bits: decodeSdxlBits(bytes, Number(width), Number(height)).map(Number).join("") }));
if (!equivalent) process.exitCode = 1;
