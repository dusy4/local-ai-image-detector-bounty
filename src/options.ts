type Settings = { enabled: boolean; minSize: number };
const DEFAULT_SETTINGS: Settings = { enabled: true, minSize: 160 };
const enabled = document.querySelector<HTMLInputElement>("#enabled")!; const minSize = document.querySelector<HTMLInputElement>("#minSize")!; const saved = document.querySelector("#saved")!;
chrome.storage.local.get(DEFAULT_SETTINGS).then(value => { const settings = value as Settings; enabled.checked = settings.enabled; minSize.value = String(settings.minSize); });
document.querySelector("#save")!.addEventListener("click", async () => { await chrome.storage.local.set({ enabled: enabled.checked, minSize: Number(minSize.value) }); saved.textContent = "Saved"; setTimeout(() => saved.textContent = "", 1200); });
