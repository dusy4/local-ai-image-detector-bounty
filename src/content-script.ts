type AnalysisResult = { score: number; label: "AI" | "real"; modelScore: number; experts?: Array<{ source: string; score: number; active: boolean; detail: string }>; signals: Array<{ source: string; detail: string }>; error?: string };
type Settings = { enabled: boolean; minSize: number };
const DEFAULT_SETTINGS: Settings = { enabled: true, minSize: 160 };

const seen = new WeakSet<HTMLImageElement>();
let settings: Settings = DEFAULT_SETTINGS;

void chrome.storage.local.get(DEFAULT_SETTINGS).then(value => { settings = value as Settings; if (settings.enabled) scan(); });
new MutationObserver(() => settings.enabled && scan()).observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("scroll", positionBadges, { passive: true });
window.addEventListener("resize", positionBadges, { passive: true });

function scan() {
  for (const image of document.images) {
    if (seen.has(image)) continue;
    seen.add(image);
    if (image.complete) void consider(image); else image.addEventListener("load", () => void consider(image), { once: true });
  }
}

async function consider(image: HTMLImageElement) {
  if (Math.min(image.naturalWidth, image.naturalHeight) < settings.minSize || !image.currentSrc) return;
  const badge = createBadge(image, "Analyzing…", "pending");
  try {
    const result = await chrome.runtime.sendMessage({ target: "service-worker", type: "analyze", url: image.currentSrc }) as AnalysisResult & { error?: string };
    if (result.error) throw new Error(result.error);
    const confidence = result.label === "AI" ? result.score : 1 - result.score;
    badge.textContent = `${result.label === "AI" ? "AI" : "Likely real"} ${Math.round(confidence * 100)}%`;
    badge.className = `local-lens-badge local-lens-${result.label === "AI" ? "ai" : "real"}`;
    badge.title = [`Local model: ${Math.round(result.modelScore * 100)}%`, ...(result.experts ?? []).map(e => `${e.source}: ${e.detail}`), ...result.signals.map(s => `${s.source}: ${s.detail}`)].join("\n");
  } catch (error) {
    badge.textContent = "Could not analyze"; badge.className = "local-lens-badge local-lens-error"; badge.title = String(error);
  }
}

function createBadge(image: HTMLImageElement, text: string, state: string) {
  const badge = document.createElement("span"); badge.textContent = text; badge.className = `local-lens-badge local-lens-${state}`;
  badge.dataset.lensFor = String(Math.random()); (badge as any)._lensImage = image; document.body.appendChild(badge); place(badge, image); return badge;
}
function positionBadges() { document.querySelectorAll<HTMLElement>(".local-lens-badge").forEach(b => place(b, (b as any)._lensImage)); }
function place(badge: HTMLElement, image: HTMLImageElement) { const r = image.getBoundingClientRect(); badge.style.left = `${scrollX + r.right - 6}px`; badge.style.top = `${scrollY + r.top + 6}px`; }
