# Baseline TensorRT Python Engine Build Report

## Executive summary

All baseline engines were built successfully using the TensorRT Python API.
Generated at: 2026-07-08T18:11:19+0000
Python interpreter: /home/user/Workspace/mergenvision/backend/.venv/bin/python
TensorRT version: 10.16.1.11

## Previous blocker summary

Previous checkpoint attempted to build with ``trtexec`` but it was missing.
``trtexec`` was not found on the host.
This checkpoint used the TensorRT Python API instead.

## Source-of-truth documents read

- AGENTS.md
- phase0beforestarting.md
- opensource/references.md
- documents/MODEL_BASELINE_DECISION.md
- documents/BASELINE_MODEL_FILES.md
- phase1/README.md
- phase1/reports/BASELINE_ONNX_INSPECTION_REPORT.md
- phase1/reports/BASELINE_ONNX_SMOKE_REPORT.md
- phase1/reports/BASELINE_TENSORRT_ENGINE_BUILD_REPORT.md

## Tool / MCP / skill usage

- TensorRT Python API used directly.
- Context7 / Exa used in previous checkpoint to verify trtexec flags; this build uses Python API.
- Old repo build scripts used as reference patterns.

## Environment snapshot

| Item | Value |
|---|---|
| Date/time | 2026-07-08T18:11:19+0000 |
| Python interpreter | /home/user/Workspace/mergenvision/backend/.venv/bin/python |
| TensorRT version | 10.16.1.11 |
| CUDA visible devices | None |
| NVIDIA-SMI parsed | no |

## Input model files

| Role | File | Path | SHA256 | Size (bytes) |
|---|---|---|---|---|
| detector | det_10g.onnx | /home/user/MergenVisionProd/artifacts/models/insightface/buffalo_l/det_10g.onnx | 5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91 | 16923827 |
| recognizer | w600k_r50.onnx | /home/user/MergenVisionProd/artifacts/models/insightface/buffalo_l/w600k_r50.onnx | 4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43 | 174383860 |

### ONNX inspection summary

- Detector: input `[1, 3, ?, ?]`, static batch=1, 9 SCRFD-style outputs.
- Recognizer: input `[None, 3, 112, 112]`, dynamic batch, output `[None, 512]`.

### ORT smoke summary

- Detector found faces with valid landmarks.
- Recognizer produced `[N, 512]` embeddings for batch=1/2/8.
- Recognizer raw outputs are not L2-normalized.

## Engine build strategy

**Detector**:
- Static batch=1 because `det_10g.onnx` has a fixed batch dimension of 1.
- Fixed 640x640 profile `min=opt=max=1x3x640x640`.
- FP16 enabled because the platform reports fast FP16 support.

**Recognizer**:
- Dynamic batch because `w600k_r50.onnx` has dynamic batch dimension.
- Profile `min=1x3x112x112`, `opt=8x3x112x112`, `max=32x3x112x112`.
- FP16 enabled because the platform reports fast FP16 support.

**Why no batch correctness is claimed yet**:
- Batch invariance, ONNX-vs-TRT numerical equivalence, and runtime inference are future checkpoints.

## Detector engine build result

- **Source ONNX**: /home/user/MergenVisionProd/artifacts/models/insightface/buffalo_l/det_10g.onnx
- **Engine path**: /home/user/MergenVisionProd/phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan
- **Input profile min/opt/max**: (1, 3, 640, 640) / (1, 3, 640, 640) / (1, 3, 640, 640)
- **Precision**: fp16
- **Build status**: pass
- **Build duration (s)**: 54.57
- **Engine size (bytes)**: 9182172
- **Engine SHA256**: 85cbfb523f67d9a65b7971239604278cd4b981c2891584aca861e05115845226
- **Log path**: /home/user/MergenVisionProd/phase1/artifacts/engine_logs/det_10g_python_trt_build.log
- **Warnings**: 0

## Recognizer engine build result

- **Source ONNX**: /home/user/MergenVisionProd/artifacts/models/insightface/buffalo_l/w600k_r50.onnx
- **Engine path**: /home/user/MergenVisionProd/phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan
- **Input profile min/opt/max**: (1, 3, 112, 112) / (8, 3, 112, 112) / (32, 3, 112, 112)
- **Precision**: fp16
- **Build status**: pass
- **Build duration (s)**: 80.578
- **Engine size (bytes)**: 88468028
- **Engine SHA256**: a1cab2f06f0dadba768df772e1342246738bbb00877710807a04c9c782f26ee6
- **Log path**: /home/user/MergenVisionProd/phase1/artifacts/engine_logs/w600k_r50_python_trt_build.log
- **Warnings**: 0

## Engine manifest

| Role | Engine path | Size | SHA256 |
|---|---|---|---|
| detector | /home/user/MergenVisionProd/phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan | 9182172 | 85cbfb523f67d9a65b7971239604278cd4b981c2891584aca861e05115845226 |
| recognizer | /home/user/MergenVisionProd/phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan | 88468028 | a1cab2f06f0dadba768df772e1342246738bbb00877710807a04c9c782f26ee6 |

## Verification commands and outputs

Run from repo root:

```bash
test -f phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan
test -f phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan
sha256sum phase1/artifacts/engines/buffalo_l/*.plan
```

## Risks and unknowns

- Engines were built with FP16 but not yet compared to ONNX Runtime output.
- Recognizer dynamic batch profile is accepted at build time but runtime behavior and batch invariance are not verified.
- Detector engine is static batch=1; multi-frame detection must loop or use separate work in the pipeline.
- License remains non-commercial/research for InsightFace pretrained weights.

## What is proven

- source_models_present_and_hashes_match: True
- tensorrt_python_imports_and_parses_models: True
- detector_engine_built: True
- recognizer_engine_built: True
- fp16_enabled_for_fast_platforms: True
- detector_static_batch_1_640x640_profile: True
- recognizer_dynamic_min1_opt8_max32_profile: True

## What is not proven yet

- tensorrt_runtime_inference
- onnx_vs_trt_numerical_equivalence
- recognizer_batch_invariance
- detector_batch_because_engine_is_static_batch_1
- fp16_accuracy_against_fp32

## Next recommended checkpoint

```text
APPROVED — START BASELINE TENSORRT SMOKE AND ONNX VS TRT COMPARISON ONLY
```

## Turkish user-facing result summary

Önceki checkpoint’te `trtexec` binary’si bu makinede bulunamadığı için bu oturumda TensorRT Python API kullanılarak engine build edildi.

**İki engine dosyası başarıyla üretildi:**
- /home/user/MergenVisionProd/phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan — 9182172 bytes
- /home/user/MergenVisionProd/phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan — 88468028 bytes

**Detector stratejisi:** `det_10g.onnx` ONNX içinde batch boyutu `1` olarak sabit olduğundan engine `batch=1` ve `640x640` profil ile üretildi. Farklı batch iddiasında bulunulmuyor.

**Recognizer stratejisi:** `w600k_r50.onnx` girdisi `[None,3,112,112]` olduğundan dinamik profile `min=1 / opt=8 / max=32` kullanıldı. Batch doğruluğu henüz kanıtlanmadı.

**Bu aşamada kanıtlananlar:** kaynak ONNX’ler mevcut, TensorRT Python parser modelleri açıyor, istenen statik/dinamik profile göre engine’ler serialize ediliyor.

**Henüz kanıtlanmayanlar:** TensorRT üzerinde runtime inference, ONNX-TRT sayısal eşdeğerlik, recognizer batch-invariance ve detector dışı batch davranışı.

**Neden LFW/benchmark yok:** bu checkpoint sadece engine üretmek içindi; karşılaştırma ve LFW sonraki checkpoint’lerde.

**Sıradaki adım:**

```text
APPROVED — START BASELINE TENSORRT SMOKE AND ONNX VS TRT COMPARISON ONLY
```
