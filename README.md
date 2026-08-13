# Local Lens

Privacy-preserving Manifest V3 Chrome extension for the [poidh local AI image detector bounty](https://poidh.xyz/arbitrum/bounty/323). It analyzes ordinary webpage images, displays a confidence score, and performs all inference inside Chrome.

## Detection pipeline

1. A bundled, quantized ViT ONNX classifier provides one real-vs-AI score.
2. Official Community Forensics and a modern-generator ConvNeXt run as active complementary experts. The locked three-model calibration scored 81.3% on held-out OpenFake and 94.6% on held-out AIGIBench rows at the fixed 0.65 threshold. FerretNet and SAFE remain research-only because they did not improve held-out accuracy.
3. The official C2PA Web SDK validates signed manifests before an AI-origin assertion becomes decisive.
4. A clean-room browser port decodes Stability AI's public 48-bit SDXL watermark.
5. An MIT-licensed carrier-phase module accepts provider/model/resolution-specific SynthID-compatible codebooks after negative-set validation.
6. Independent validated signals are fused monotonically. The required decision threshold is fixed at 65%.

`dev/synthid-trainer.html` is an independent MIT implementation of the reproducible method: it discovers coherent carrier frequencies and phases from user-supplied provider outputs, calibrates them against negatives, and exports the compact codebook consumed by the extension.

## Reproducible build

Requirements: Node.js 22+, npm, and internet access only while building.

```powershell
npm.cmd run model:download
npm.cmd run check
```

Load `dist` using `chrome://extensions` → Developer mode → Load unpacked. The model is packaged in `dist`; disconnecting the network after installation does not affect inference.

## Benchmark

Place held-out images under `benchmark/real` and `benchmark/ai`, then run `npm.cmd run benchmark`. The output reports TPR, TNR, and balanced accuracy at exactly 0.65. Do not tune on the bounty's private evaluation data.

Current evidence is recorded under `reports/`. The original classifier alone scored 76.0% on 200 OpenFake images but only 54.0% on an independent 300-image AIGIBench sample. Community Forensics scored 62.5% and 92.0% respectively. The quantized modern ConvNeXt scored 79.5% and 61.7%, and caught 5/5 Flux Krea samples missed by Community Forensics. The locked three-model calibration scored 81.3% and 94.6% on source-stratified held-out halves. These are public directional results, not the private bounty score.

On 250 web-transformed sibling variants, the locked pipeline scored 84.4% balanced accuracy; JPEG, resize, crop, blur, and contrast groups each scored at least 80.0%. A fresh-profile smoke also succeeds after Chrome is switched offline.

Runtime inference is serialized to avoid loading several large WASM graphs concurrently, and duplicate image URLs share an in-flight result plus a bounded local cache. A three-image duplicate smoke completes with three labels from one analysis pass.

## Submission gate

- At least 75% balanced accuracy at threshold 0.65 on a web-realistic held-out set.
- Per-source results for real photos, illustrations, screenshots, JPEG recompression, crops, and current generators.
- Clean-profile installation and offline test.
- Model and SynthID codebook hashes recorded and reproducible.
- No network requests at runtime; verify in Chrome DevTools after initial installation.

## License

MIT.
