# AI-image detection research for bounty 323

Research snapshot: 2026-08-13. The target is a Manifest V3 Chrome extension that runs fully in-browser and is judged at a fixed 0.65 threshold.

## Integration results

- FerretNet-B was exported from the pinned official checkpoint to a 4.23 MB ONNX graph. Its masked-median residual is reproduced in browser JavaScript. The graph runs under ONNX Runtime Web, but the initial 200-image OpenFake suite produced 1% AI recall / 99% real recall at 0.65, or 50.0% balanced accuracy. It therefore remains visible in shadow mode and cannot affect the verdict.
- SAFE was exported from the pinned official checkpoint to a 5.74 MB ONNX graph. Its browser bior1.3 diagonal-band transform matches the official PyTorch preprocessing to a maximum absolute error of `4.47e-8`; the graph runs under ONNX Runtime Web. On the initial 200-image suite its native 0.65 score produced 0% AI recall / 100% real recall (50.0% balanced accuracy); even the best in-sample threshold reached only 60.0%, so it remains excluded from voting.
- The official Community Forensics 224px checkpoint was exported to an 86.8 MB ONNX graph and runs in browser WASM. It scored 62.5% on 200 OpenFake images but 92.0% on an independent 300-image AIGIBench sample. The threshold-OR hybrid with the original classifier scored 81.0% and 77.0% respectively, so Community Forensics is active.
- A modern-generator ConvNeXtV2-Base checkpoint trained on Flux, Nano Banana Pro, DALL-E, Midjourney and Stable Diffusion families was exported and linear-weight quantized to a 100.4 MB browser-compatible graph. It scored 79.5% on OpenFake, 61.7% on AIGIBench, and detected 5/5 Flux Krea samples that Community Forensics missed. Source-stratified cross-dataset calibration of all three active models held out at 81.3% and 94.6% respectively.
- The Apache-2.0 `ash12321/flux-detector-vit` specialist was independently tested and rejected. At the bounty threshold it detected only 1/25 Flux 2 Klein images, 0/10 GPT Image 1.5 images, 4/7 Midjourney v7 images, and 0/2 Recraft v3 images while correctly retaining 45/50 real images. Its published FLUX.1-dev result does not transfer to these newer providers, so adding its 343 MB checkpoint would reduce efficiency without closing the main recall gaps. Reproduction code and per-image scores are in `scripts/benchmark-flux-vit.py` and `reports/flux-vit-providers.json`.
- Six-crop max TTA, independently reproduced from the public Caravela claim using our active Community Forensics graph, raised the calibrated OpenFake result from 80.0% to 81.5% balanced accuracy but reduced specificity from 95% to 91%. On a separate 100-image AIGIBench slice it raised the same ensemble from 86% to 87%. Applying TTA only when the center-crop ensemble lands in the 60–65% borderline band improved those results to 82.5% and 87%, respectively, with far fewer extra inferences. The extension therefore uses selective—not unconditional—max-6 TTA.
- A clean-room RGB residual phase-coherence experiment trained on five verified low-entropy OpenAI outputs did not transfer to three held-out OpenAI positives: positive mean match was 0.493 versus 0.502 across 100 real negatives. The experiment is retained as falsifying evidence; no spectral codebook was activated.
- `@contentauth/c2pa-web` 0.13.4 is bundled locally. A clean browser run parsed the C2PA public test fixture as `Valid`; raw assertions and merely valid self-signed manifests cannot vote. Only `Trusted` signer status plus an AI-origin assertion is decisive.
- The Stability AI SDXL 48-bit DWT/DCT decoder is integrated. On the supplied Gemini image both the official Python decoder and browser port report no watermark (25 matching bits). On a positive fixture they report 46 and 47 matching bits respectively, both far over the official likely-watermarked boundary. The official decoder produced zero false positives across 197 eligible OpenFake benchmark images; the maximum negative match was 30/48 versus the 34-bit activation boundary.
- A fresh headless Chrome profile successfully loaded the unpacked extension and labeled the supplied Gemini image `AI 65%`; FerretNet and SAFE scores appeared independently in the badge diagnostics.
- Three fresh, project-owned OpenAI images spanning photography, illustration, and product imagery were generated and tested on 2026-08-13. OpenAI's official verifier identified all three as generated with OpenAI tools; detail inspection on one reported `SynthID detected` and `Content Credentials not detected`. The local ensemble detected 2/3 at 0.65, although the active modern ConvNeXt scored above 0.97 on all three. Five additional low-entropy OpenAI outputs brought ensemble recall to 6/8. This confirms that an OpenAI-specific SynthID decoder is the most valuable missing expert and that changing fusion from this tiny selected set would be overfitting.

## Decision

Do not build the detector from scratch and do not make carrier-phase detection the primary detector.

The strongest practical design is a calibrated mixture of independent signals:

1. **Community Forensics** as the broad semantic/generator-diversity expert.
2. **FerretNet** and **SAFE** as tiny low-level forensic experts.
3. Cryptographically validated **C2PA** plus conventional generator metadata.
4. Exact provider watermark decoders where the key/decoder is public, starting with Stability AI's SDXL DWT/DCT watermark.
5. Provider-specific spectral codebooks only when trained and validated on that exact provider/model/resolution.

The current face-oriented ViT is only a placeholder. Its model card says it was trained on real and fake human faces and explicitly warns that it may not generalize to other domains. That matches our narrow OpenFake result: 76% balanced accuracy but only 68% AI recall.

## What the evidence says

The central problem is unseen-generator generalization, not merely model architecture.

- A February 2026 study evaluated 23 released detector variants across 12 datasets and 291 generators. The best detector, Community Forensics, averaged only 75.0% accuracy; detector rankings were unstable across datasets, and modern Flux Dev, Firefly v4, and Midjourney v7 images averaged only 18–30% detection across methods. Training-data alignment caused 20–60 percentage-point differences within the same architecture. [Paper](https://arxiv.org/abs/2602.07814)
- Community Forensics trained on 2.7 million images from 4,803 generator models. Its core finding is that performance improves with generator diversity. [Paper](https://arxiv.org/abs/2411.04125) · [official code](https://github.com/JeongsooP/Community-Forensics)
- AIGIBench's unified retraining comparison shows the same imbalance: many methods retain high real-image accuracy but miss synthetic images. On its broad test set, SAFE scores 89.0% real / 66.6% fake, AIDE 88.1% / 67.0%, and FerretNet 96.6% / 61.8%. [Official benchmark](https://github.com/HorizonTEL/AIGIBench)
- RAID provides 72,000 transferable adversarial examples and shows that current detectors are easily deceived. [Paper](https://arxiv.org/abs/2506.03988)

Therefore a single published accuracy number is not enough. We need source-separated testing, current providers, web transformations, and probability calibration at exactly 0.65.

## Public detector candidates

| Candidate | Public evidence | Browser fit | License / availability | Verdict |
|---|---|---:|---|---|
| Community Forensics | Best mean zero-shot result, 75.0%, in the 2026 12-dataset study; trained across 4,803 models | ViT-Small, 384 px; 143.8 MB fp32; public q4 is 24.4 MB | Official code MIT; community Transformers.js conversion MIT | **Primary broad expert. Re-export ourselves.** |
| FerretNet | 1.06M parameters; official paper reports 97.1% average on four established test groups; AIGIBench reports 79.4% overall / 85.8 AP | About 4.3 MB weights; simple depthwise CNN; 256 px | Apache-2.0; weights are in GitHub | **Best first browser port.** |
| SAFE | 1.44M parameters; AIGIBench reports 78.1% overall; official repo reports 96–99% detection on two GPT-4o subsets | About 5.8 MB; truncated ResNet plus DWT high-frequency preprocessing; 256 px | Apache-2.0; weights are in GitHub | **Best complementary wavelet expert.** |
| AIDE | Hybrid CLIP semantic features plus selected high/low-frequency patches; stronger than older baselines on Chameleon, GenImage, and AIGCDetectBenchmark | ConvNeXt/CLIP hybrid is much larger and harder to export | Official code available; verify all dependency/weight licenses | Useful research reference; too heavy for first port |
| UniversalFakeDetect / CLIP linear head | Strong historical cross-generator baseline and robust to some laundering | CLIP ViT-L/14 is hundreds of MB; quantization possible | Official code available | Useful semantic fallback, but Community Forensics is newer and stronger |
| GAPL | Generator-aware prototypes plus LoRA; CVPR 2026 and public weights | CLIP-based and substantially heavier | Official code/weights; repository license must be verified before shipping | Benchmark after lightweight experts |
| TAP with modern foundation models | 2026 paper reports over 12 points above original CLIP and strong patch-token pooling | Latest backbones are likely too large for an extension | Paper is public; deployment artifacts not yet established | Future candidate, not immediate |
| Forensic Self-Descriptions | 96% average AUC across 24 generators while trained only on real images | Roughly 55 MB detection extras plus feature extractor; conversion work | CC BY-NC-SA, research-only | Do not ship in MIT bounty entry |

### Empirical browser compatibility result

The public `onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX` export was tested in the same Transformers.js/WASM runtime used by the extension:

- q4 failed during browser inference with a WASM runtime error.
- int8 could not create a session because `ConvInteger(10)` had no WASM implementation.
- fp32 also failed during WASM session creation, so merely accepting the 143.8 MB download does not fix deployment.
- The candidate's model card claims Transformers.js support, but those quantized artifacts are not deployable as-is in this extension.

This does not invalidate Community Forensics. It means we should export the official PyTorch checkpoint ourselves and quantize weights while keeping the patch-embedding convolution in a browser-supported format, then test the resulting graph directly with ONNX Runtime Web before integration.

## Why FerretNet should be ported first

FerretNet is almost purpose-built for this bounty:

- 1.06M parameters and a 4.29 MB checkpoint.
- Apache-2.0 code and bundled official weights.
- Simple convolution, batch normalization, ReLU, pooling, and a two-class head.
- Uses local pixel dependency residuals: input minus a 3×3 median neighborhood estimate. This catches excessive smoothing and local texture discontinuities left by decoders.
- The median preprocessing can be implemented deterministically in JavaScript and excluded from ONNX, avoiding unsupported median/unfold operators.

Its weakness is exactly what the ensemble fixes: on AIGIBench it has excellent real-image specificity but weaker AI recall. It should contribute a forensic likelihood, not make the final decision alone.

## Why SAFE is complementary

SAFE uses only the early half of a ResNet-like network after extracting a high-frequency wavelet band. Its official checkpoint is about 5.8 MB. The transform and cue differ from FerretNet:

- FerretNet: local median residual / pixel dependency.
- SAFE: DWT high-frequency diagonal band / transformation artifacts.
- Community Forensics: learned broad generator distribution.

That diversity is more useful than stacking three similar ViTs. SAFE's published GPT-4o-positive results are promising, but they contain no matched real negatives in those subsets, so they do not establish balanced accuracy by themselves.

## Provider-specific signals

### C2PA and metadata

Use the official `@contentauth/c2pa-web` WASM library, bundled locally, to parse and validate manifests. The current lightweight byte-string parser is useful as a fallback but does not cryptographically validate the signer.

High-confidence AI evidence includes a valid trusted manifest with actions or digital-source types indicating trained algorithmic media. Generic C2PA must not be treated as AI because cameras and news organizations also sign authentic images.

The official browser SDK is MIT, runs fully client-side, and supports a separately hosted local WASM binary. Bundle a pinned trust-list snapshot for offline use after setup.

### OpenAI and Google SynthID

OpenAI officially added SynthID alongside C2PA on 2026-05-19 for images generated by ChatGPT, Codex, and its API. Google's and OpenAI's production detectors remain proprietary. Their public verification pages cannot be used because the bounty forbids cloud/API inference.

The spectral carrier approach is useful as a research recipe, not as a universal key. Our initial codebook test did not detect either the supplied Gemini image or a fresh OpenAI image. The likely causes are provider/model/resolution-specific carrier selection and changed watermark versions.

Keep the independent FFT phase-coherence implementation, but only enable a codebook after:

1. collecting 50–200 solid/low-entropy outputs per provider, model, color family, and resolution;
2. extracting consistent carrier phase across positives;
3. validating against at least 1,000 unrelated real and synthetic negatives;
4. proving that its 65% score has a controlled false-positive rate.

### Stability AI / SDXL

This is the easiest exact public watermark win. The official Stability AI and Diffusers implementations use `invisible-watermark` DWT+DCT with a fixed 48-bit message:

`0xB3EC907BB19E`

The official detector reports:

- fewer than 27 matching bits: no watermark;
- 27–32: partial match;
- 33–34: likely match with a reported 0.02% real-image false-positive rate at the 35-bit boundary;
- 35+ matched bits: very likely watermarked in its published test.

Port the decoder to browser JavaScript and treat a strong match as provider-specific high-confidence evidence. Coverage is limited because many forks disable watermarking and newer Stability systems may differ.

### Other open watermark systems

Watermark Anything, Stable Signature, DistSeal, and related Meta systems publish decoders and some MIT weights. They only help if a provider actually embeds the corresponding watermark/message. Do not run every decoder blindly: calibrate each against its known message and a large negative set, then enable it as a sparse expert.

## Benchmark data to use

The existing first-N OpenFake sample is directional, not enough for model selection. Build a deterministic, source-balanced suite from:

1. **T2I-CoReBench-Images**: 172,800 generated images from 40 current models, including Nano Banana, Imagen 4, GPT-Image, Seedream, Flux 2, Qwen Image, and autoregressive/unified models. Use this for current synthetic coverage.
2. **AIGIBench**: balanced real/fake subsets spanning Midjourney v6, SD3, Imagen, DALL-E 3, Flux, community, and social images.
3. **Community Forensics test split**: broad source diversity and thousands of generator models.
4. **OpenFake/WildFake/SynthBuster**: historical and in-the-wild cross-checks.
5. A real-only negative suite: camera photos, scans, screenshots, charts, illustrations, 3D renders, heavily edited photos, and ordinary web graphics.

For each untouched source image create sibling test variants, never split siblings across calibration and test:

- JPEG 95/80/60;
- WebP;
- resize to 256, 512, and 1024;
- center and random crops;
- screenshot/re-encode;
- light blur, sharpen, color, and contrast changes.

Report balanced accuracy, TPR, TNR, AUROC, calibration error, 95% bootstrap intervals, per-generator results, and per-transformation results. The final test gate should be at least 80% balanced accuracy so sampling variance is unlikely to pull the private score below 75%.

## Implementation order

1. Export and integrate FerretNet with exact preprocessing; calibrate at 0.65.
2. Export and integrate SAFE; measure whether score fusion improves held-out AI recall without losing real specificity.
3. Re-export Community Forensics into browser-compatible ONNX and benchmark it alone and in the ensemble.
4. Replace heuristic C2PA parsing with the official offline browser verifier.
5. Add the SDXL fixed-message DWT/DCT decoder.
6. Train provider-specific SynthID spectral codebooks only where data proves value.
7. Lock calibration, run clean-profile/offline extension QA, then publish and submit.

## Stop/go rules

- Do not ship a candidate because its paper accuracy is high; require improvement on our source-separated held-out set.
- Do not fuse raw probabilities from separately calibrated models. Fit a small logistic/temperature calibration layer on model logits and binary provenance signals.
- Do not allow weak watermark correlation to raise a score above 0.65.
- Do allow cryptographically valid AI C2PA and statistically validated exact watermark matches to override a weak generic classifier.
- Drop any expert that adds latency but does not improve the lower confidence bound of balanced accuracy.
