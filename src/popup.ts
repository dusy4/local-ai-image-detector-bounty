const status = document.querySelector("#status")!;
chrome.runtime.sendMessage({ target: "service-worker", type: "health" }).then(result => { status.textContent = result?.ready ? "Detector ready" : "Detector unavailable"; }).catch(() => { status.textContent = "Detector starts on first image"; });
document.querySelector("#options")!.addEventListener("click", () => chrome.runtime.openOptionsPage());
