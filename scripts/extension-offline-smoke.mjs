import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const extension = resolve("dist");
const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  pipe: true,
  enableExtensions: [extension],
  userDataDir: resolve(".chrome-offline-smoke"),
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  const runtimeErrors = [];
  page.on("pageerror", error => runtimeErrors.push(String(error)));
  await page.goto("http://127.0.0.1:4173/dev/offline-smoke.html", { waitUntil: "networkidle0", timeout: 30_000 });
  await page.evaluate(() => window.__prepared);
  await page.setOfflineMode(true);
  await page.evaluate(() => window.__appendPrepared());
  await page.waitForSelector(".local-lens-badge:not(.local-lens-pending)", { timeout: 120_000 });
  const badge = await page.$eval(".local-lens-badge", element => ({ text: element.textContent, title: element.getAttribute("title") }));
  console.log(JSON.stringify({ offline: true, badge, runtimeErrors }));
  if (runtimeErrors.length || !badge.text?.includes("AI")) process.exitCode = 1;
} finally {
  await browser.close();
}
