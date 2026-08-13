import puppeteer from "puppeteer-core";

const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("console", message => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", error => errors.push(String(error)));
  await page.goto(process.argv[2], { waitUntil: "networkidle0", timeout: 30000 });
  await page.waitForFunction(() => document.querySelector("#result")?.textContent !== "pending", { timeout: 30000 });
  const result = await page.$eval("#result", element => element.textContent);
  console.log(JSON.stringify({ result, errors }));
  if (result?.startsWith("ERROR") || errors.length) process.exitCode = 1;
} finally {
  await browser.close();
}
