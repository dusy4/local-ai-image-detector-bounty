# Bounty execution plan

## 1. Baseline and evidence

- Build a balanced public benchmark with strict source-level separation: camera photos, web screenshots/illustrations, public benchmark AI samples, and 2025–2026 generators.
- Preserve untouched originals plus web-realistic JPEG/WebP, resize, crop, and screenshot variants.
- Record provenance; never train and test on sibling variants of the same source image.

## 2. Detector candidates

- [done] Port FerretNet (Apache-2.0, 1.06M parameters) with exact local-median residual preprocessing. It runs in browser shadow mode because the first held-out suite scored 50.0% balanced accuracy.
- [done] Port SAFE (Apache-2.0, 1.44M parameters) with exact bior1.3 DWT preprocessing. Browser graph and preprocessing parity are verified. Its native threshold scored 50.0% balanced accuracy and the best in-sample threshold scored 60.0%, so it remains excluded from voting.
- [done] Re-export the official MIT Community Forensics 224px checkpoint as a browser-compatible 86.8 MB ONNX graph. A monotonic threshold-OR hybrid with the original classifier scored 81.0% on OpenFake and 77.0% on an independent AIGIBench sample, so Community Forensics is active.
- Retain the bundled Deep-Fake-Detector-v2 only as a benchmark baseline; its own model card says it is face-domain-specific.
- Select or ensemble only candidates that improve held-out balanced accuracy at the bounty's fixed 65% threshold.

## 3. Provenance and watermarks

- [done] Parse generator metadata without treating generic camera Content Credentials as AI.
- [done] Replace heuristic C2PA trust with pinned `@contentauth/c2pa-web`; raw strings and merely valid self-signed manifests are non-voting. A public valid manifest passes browser verification. Only `Trusted` signer status plus AI origin can vote; provider trust-list coverage still needs explicit testing.
- [done] Port Stability AI's public 48-bit SDXL DWT/DCT watermark decoder. The browser decoder agrees with the upstream implementation on negative and positive fixtures, the installed extension detects a positive fixture, and the reference decoder produced 0/197 false positives on the current suite.
- Derive provider-specific codebooks from reference outputs, then validate on Gemini/OpenAI positives and at least 1,000 unrelated negatives.
- Add Stable Signature or legacy DWT/DCT decoders only with public keys, compatible weights, and measured false-positive bounds.

## 4. Calibration

- Collect current-provider outputs under `provider-input/ai/<provider>` and owned camera/reference images under `provider-input/real/<source>`. `npm run calibration:prepare` hashes every file and fails on duplicates or overlap with the locked OpenFake test set.
- Fit temperature/bias on a calibration split, lock parameters, then report the untouched test split.
- Optimize calibration at the fixed 0.65 decision threshold rather than changing the threshold.
- Require bootstrap confidence intervals and publish the confusion matrix by generator, source, and transformation.

## 5. Bounty submission

- Verify `npm run model:download && npm run check` from a clean checkout.
- Load into a fresh Chrome profile; block internet; test ordinary pages and dynamic images.
- Record a short demo and submit the public GitHub repository as the earliest valid claim only after the ≥75% gate is independently reproduced.

## Current evidence

- 100-image OpenFake held-out sample: 78.0% balanced accuracy at 0.65.
- Expanded 200-image sample: 76.0% balanced accuracy at 0.65 (68% TPR, 84% TNR).
- Independent 300-image AIGIBench sample: original classifier 54.0%, Community Forensics 92.0%, active threshold-OR hybrid 77.0% balanced accuracy at 0.65.
- Modern-generator ConvNeXt: 79.5% OpenFake, 61.7% AIGIBench, and 5/5 Flux Krea recall at 0.65 after browser-compatible weight quantization.
- Locked three-model calibration on source-stratified held-out halves: 81.3% OpenFake and 94.6% AIGIBench. The installed extension passes Flux Krea AI and AIGIBench real end-to-end smokes.
- Web transformations: 84.4% balanced accuracy across 250 JPEG, resize, crop, blur, and contrast variants; every transform group scored at least 80.0%.
- Selective six-crop Community Forensics TTA for borderline 60–65% cases raises the current OpenFake evidence to 82.5% and a separate 100-image AIGIBench slice to 87.0%, while avoiding the false-positive and latency cost of unconditional TTA.
- A staged clean checkout reproduces `npm ci`, pinned baseline model download, tests, and build. A fresh Chrome profile detects the supplied Gemini image after network access is disabled.
- The ensemble clears the current public gates but remains narrow. Increase current commercial-generator coverage before claiming.
- Three fresh OpenAI images generated on 2026-08-13 exposed the highest-priority gap: the current ensemble detected 2/3 at 0.65, while the modern ConvNeXt expert detected 3/3. OpenAI's official verifier detected the generated origin in all three and reported SynthID with no trusted C2PA manifest in the inspected file. Five additional low-entropy outputs brought local ensemble recall to 6/8. Do not retune on this tiny holdout; obtain a properly sized, source-separated set or implement the watermark decoder.
- Deep research and ranked candidates are documented in `RESEARCH.md`.
- The installed extension now passes a clean headless-Chrome end-to-end smoke test on the supplied Gemini image with both forensic experts present.
