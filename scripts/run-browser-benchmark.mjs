import { writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import puppeteer from "puppeteer-core";

let server;
const outputPath = process.argv[2] ?? "reports/openfake-baseline-200-rows.json";
const manifestPath = process.argv[3] ?? "/benchmark/manifest.json";
const limitPerClass = Number(process.argv[4] ?? 0);
try { await fetch(`http://127.0.0.1:4173${manifestPath}`); }
catch {
  server = spawn(process.execPath, ["scripts/serve.mjs"], { stdio: "ignore", windowsHide: true });
  for (let attempt = 0; attempt < 50; attempt++) {
    try { if ((await fetch("http://127.0.0.1:4173/benchmark/manifest.json")).ok) break; } catch {}
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

const browser = await puppeteer.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe", headless: true, protocolTimeout: 1_500_000, args: ["--no-sandbox"] });
try {
  const page = await browser.newPage();
  const benchmarkUrl = `http://127.0.0.1:4173/dev/benchmark.html?manifest=${encodeURIComponent(manifestPath)}&limitPerClass=${limitPerClass}`;
  await page.goto(benchmarkUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => document.title.startsWith("DONE") || document.title === "ERROR", { timeout: 1_200_000 });
  if (await page.title() === "ERROR") throw new Error(await page.$eval("#result", element => element.textContent));
  const report = await page.evaluate(() => window.__benchmarkReport);
  await writeFile(outputPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ images: report.images, tpr: report.tpr, tnr: report.tnr, balancedAccuracy: report.balancedAccuracy }));
} finally {
  await browser.close();
  server?.kill();
}
