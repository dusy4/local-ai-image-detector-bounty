const OFFSCREEN_PATH = "offscreen.html";
let creating: Promise<void> | undefined;
let analysisTail: Promise<void> = Promise.resolve();
const resultCache = new Map<string, unknown>();
const inFlight = new Map<string, Promise<unknown>>();
const CACHE_LIMIT = 128;

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "service-worker") return;
  route(message).then(sendResponse).catch(error => sendResponse({ error: String(error?.message ?? error) }));
  return true;
});

async function route(message: { type: string; url?: string }) {
  if (message.type === "analyze" && message.url) {
    const cached = resultCache.get(message.url);
    if (cached) return cached;
    const existing = inFlight.get(message.url);
    if (existing) return existing;
    const task = analysisTail.then(() => routeUnqueued(message));
    analysisTail = task.then(() => undefined, () => undefined);
    inFlight.set(message.url, task);
    try {
      const result = await task;
      resultCache.set(message.url, result);
      if (resultCache.size > CACHE_LIMIT) resultCache.delete(resultCache.keys().next().value!);
      return result;
    } finally {
      inFlight.delete(message.url);
    }
  }
  return routeUnqueued(message);
}

async function routeUnqueued(message: { type: string; url?: string }) {
  await ensureOffscreen();
  await waitForOffscreen();
  return chrome.runtime.sendMessage({ ...message, target: "offscreen" });
}

async function waitForOffscreen(): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      const response = await chrome.runtime.sendMessage({ target: "offscreen", type: "health" });
      if (response?.ready) return;
    } catch { /* listener is still loading */ }
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error("Detector initialization timed out");
}

async function ensureOffscreen(): Promise<void> {
  const url = chrome.runtime.getURL(OFFSCREEN_PATH);
  const contexts = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"], documentUrls: [url] });
  if (contexts.length) return;
  creating ??= chrome.offscreen.createDocument({ url: OFFSCREEN_PATH, reasons: ["WORKERS"], justification: "Run private local image inference outside webpage contexts" }).finally(() => { creating = undefined; });
  await creating;
}
