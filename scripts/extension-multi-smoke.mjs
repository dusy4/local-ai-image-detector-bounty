import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  pipe: true,
  enableExtensions: [resolve("dist")],
  args: ["--no-sandbox"],
});
const started = performance.now();
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  await page.goto("http://127.0.0.1:4173/dev/extension-multi-smoke.html", { waitUntil: "networkidle0", timeout: 30_000 });
  await page.waitForFunction(() => document.querySelectorAll(".local-lens-badge:not(.local-lens-pending)").length === 3, { timeout: 120_000 });
  const badges = await page.$$eval(".local-lens-badge", elements => elements.map(element => element.textContent));
  const report = { badges, elapsedMs: Math.round(performance.now() - started), errors };
  console.log(JSON.stringify(report));
  if (errors.length || badges.length !== 3 || badges.some(text => !text?.startsWith("AI"))) process.exitCode = 1;
} finally {
  await browser.close();
}
