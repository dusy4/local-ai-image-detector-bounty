# Third-party notices

- `@huggingface/transformers`: Apache-2.0.
- `onnx-community/Deep-Fake-Detector-v2-Model-ONNX`: Apache-2.0 per its model card. The downloaded artifact is pinned and hashed by `model-lock.json`.
- `onnxruntime-web` 1.22.0: MIT. The bundled runtime is pinned and hashed by `runtime-lock.json`.
- `xigua7105/FerretNet`: Apache-2.0. The official checkpoint and our ONNX export are pinned and hashed by `ferretnet-lock.json`.
- `@contentauth/c2pa-web` 0.13.4 and its dependencies: MIT. Bundled locally for cryptographic C2PA manifest validation; its worker URL guard is minimally patched to permit Chrome extension URLs while retaining HTTPS-only web URLs.
- `Ouxiang-Li/SAFE`: Apache-2.0. The official checkpoint and our ONNX export are pinned and hashed by `safe-lock.json`.
- `JeongsooP/Community-Forensics` and `OwensLab/commfor-model-224`: MIT. The official checkpoint and our ONNX export are pinned and hashed by `community-forensics-lock.json`.
- `xRayon/convnext-ai-images-detector`: MIT. The pinned phase-two checkpoint is exported to ONNX and linear weights are quantized to int8; the shipped graph is pinned and hashed by `modern-convnext-lock.json`.
