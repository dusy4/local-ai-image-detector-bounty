import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const extension = resolve("dist");
const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  pipe: true,
  enableExtensions: [extension],
  args: ["--no-sandbox"],
});
try {
  const workerErrors = [];
  browser.on("targetcreated", async target => {
    if (target.type() !== "service_worker") return;
    const worker = await target.worker();
    worker?.on("console", message => workerErrors.push(message.text()));
  });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  await page.goto(process.argv[2] ?? "http://127.0.0.1:4173/dev/extension-smoke.html", { waitUntil: "networkidle0", timeout: 30000 });
  try { await page.waitForSelector(".local-lens-badge:not(.local-lens-pending)", { timeout: 90_000 }); } catch {}
  const badge = await page.$eval(".local-lens-badge", element => ({ text: element.textContent, title: element.getAttribute("title") })).catch(() => null);
  console.log(JSON.stringify({ badge, errors, workerErrors, targets: browser.targets().map(target => `${target.type()}:${target.url()}`) }));
  const expected = process.argv[3] ?? "AI";
  if (errors.length || workerErrors.length || !`${badge?.text}\n${badge?.title}`.includes(expected)) process.exitCode = 1;
} finally {
  await browser.close();
}
