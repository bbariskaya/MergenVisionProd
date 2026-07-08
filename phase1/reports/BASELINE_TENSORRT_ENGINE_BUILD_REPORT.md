# Baseline TensorRT Engine Build Report

## 1. Executive summary

This checkpoint attempted to build TensorRT `.plan` engines from the already downloaded InsightFace `buffalo_l` baseline ONNX files:

- `det_10g.onnx` (SCRFD-10GF detector)
- `w600k_r50.onnx` (ArcFace R50 recognizer)

The build **did not run** because the `trtexec` command-line tool is not installed on this host. TensorRT Python bindings are available inside an existing old-project virtual environment (`/home/user/Workspace/mergenvision/backend/.venv`, TensorRT 10.16.1.11), but the CLI tool required by this checkpoint's scope is absent. No engine files were produced.

Status: **BLOCKED** — environment lacks `trtexec`, not a model or build logic failure.

---

## 2. Source-of-truth documents read

- `/home/user/MergenVisionProd/AGENTS.md`
- `/home/user/MergenVisionProd/phase0beforestarting.md`
- `/home/user/MergenVisionProd/opensource/references.md`
- `/home/user/MergenVisionProd/documents/MODEL_BASELINE_DECISION.md`
- `/home/user/MergenVisionProd/documents/BASELINE_MODEL_FILES.md`
- `/home/user/MergenVisionProd/phase1/README.md`
- `/home/user/MergenVisionProd/phase1/reports/BASELINE_ONNX_INSPECTION_REPORT.md`
- `/home/user/MergenVisionProd/phase1/reports/baseline_onnx_inspection.json`
- `/home/user/MergenVisionProd/phase1/reports/BASELINE_ONNX_SMOKE_REPORT.md`
- `/home/user/MergenVisionProd/phase1/reports/baseline_onnx_smoke.json`

Old/repo reference files inspected (read-only):

- `/home/user/Workspace/mergenvision/backend/scripts/build_trt_engines.py`
- `/home/user/MergenVision/backend/scripts/build_trt_engines.py`
- `/home/user/Workspace/MergenVisionCleanVersion/TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md`

---

## 3. Tool / MCP / skill usage

- **codebase-memory-mcp**: checked index status; `MergenVisionProd` is not indexed. Used filesystem reads instead.
- **deepwiki**: skipped; filesystem already contained the relevant old-repo build scripts and the TensorRT strategy document.
- **exa / web search**: used to verify current `trtexec` dynamic-shape flag syntax.
- **context7**: used to verify official TensorRT dynamic-shape optimization profile and FP16 engine build documentation.
- **shell/filesystem**: used for environment checks, locating `trtexec`, inspecting venvs, reading docs, checking disk space.

Skills used:

- `executing-plans`
- `systematic-debugging`
- `verification-before-completion`
- `self-review/code-review`

---

## 4. Environment snapshot

| Item | Value |
|---|---|
| Date/time | 2026-07-08T18:04:00+00:00 |
| Host | ubuntu |
| GPU name | Quadro RTX 8000 (3x present) |
| Driver version | 580.105.08 |
| CUDA version | 13.0 |
| TensorRT Python version | 10.16.1.11 (only in `/home/user/Workspace/mergenvision/backend/.venv`) |
| `trtexec` path | **not found** |
| Docker version | 29.1.5 |
| Disk available | 983 GB |

GPU state at check: all three GPUs idle; no running processes.

---

## 5. Input model files

| Role | File | Local path | SHA256 | Size |
|---|---|---|---|---|
| Detector | `det_10g.onnx` | `artifacts/models/insightface/buffalo_l/det_10g.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | 16.9 MB |
| Recognizer | `w600k_r50.onnx` | `artifacts/models/insightface/buffalo_l/w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` | 174.4 MB |

### ONNX inspection summary

- **Detector**: input shape `[1, 3, ?, ?]`, static batch=1, 9 SCRFD-style outputs (score/bbox/kps × stride 8/16/32). Standard input 640x640.
- **Recognizer**: input shape `[None, 3, 112, 112]`, dynamic batch, output `[None, 512]`.

### ORT smoke summary

- Detector found faces in all 3 LFW samples with valid 5-point landmarks (scores ~0.81–0.85).
- Recognizer produced `[N, 512]` embeddings for batch=1/2/8 with perfect batch invariance.
- Raw recognizer outputs are **not** L2-normalized; production pipeline must normalize after inference.

---

## 6. Engine build strategy

### Detector

- **Why batch=1**: `det_10g.onnx` has a static batch dimension of 1 in the ONNX graph. We do not patch or re-export ONNX.
- **Profile**: `min=opt=max=1x3x640x640`.
- **Output name**: `det_10g_b1_640_fp16.plan`.

### Recognizer

- **Why dynamic profile**: `w600k_r50.onnx` has a dynamic batch dimension (`None`).
- **Profile**: `min=1x3x112x112`, `opt=8x3x112x112`, `max=32x3x112x112`.
- **Output name**: `w600k_r50_min1_opt8_max32_fp16.plan`.

### Why no batch correctness is claimed yet

Batch invariance under TensorRT, ONNX-vs-TRT numerical equivalence, and dynamic-batch runtime behavior are **future checkpoints**. This checkpoint was intended only to produce engines.

---

## 7. Detector engine build

**Status**: NOT STARTED (blocked by missing `trtexec`).

Intended command:

```text
trtexec \
  --onnx=artifacts/models/insightface/buffalo_l/det_10g.onnx \
  --saveEngine=phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan \
  --fp16 \
  --minShapes=input.1:1x3x640x640 \
  --optShapes=input.1:1x3x640x640 \
  --maxShapes=input.1:1x3x640x640 \
  --timingCacheFile=phase1/artifacts/timing_cache/buffalo_l.cache
```

Intended output: `phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan`

Actual output: none.

---

## 8. Recognizer engine build

**Status**: NOT STARTED (blocked by missing `trtexec`).

Intended command:

```text
trtexec \
  --onnx=artifacts/models/insightface/buffalo_l/w600k_r50.onnx \
  --saveEngine=phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan \
  --fp16 \
  --minShapes=input.1:1x3x112x112 \
  --optShapes=input.1:8x3x112x112 \
  --maxShapes=input.1:32x3x112x112 \
  --timingCacheFile=phase1/artifacts/timing_cache/buffalo_l.cache
```

Intended output: `phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan`

Actual output: none.

---

## 9. Engine manifest

No engine files were produced. Intended manifest entries:

| Engine | Path | Status |
|---|---|---|
| Detector FP16 batch=1 640x640 | `phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan` | not built |
| Recognizer FP16 dynamic batch 1/8/32 | `phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan` | not built |

---

## 10. Verification commands and outputs

```text
cd /home/user/MergenVisionProd
nvidia-smi
# Driver 580.105.08, CUDA 13.0, 3x Quadro RTX 8000, all idle.

which trtexec
# (empty)

find /usr -name trtexec 2>/dev/null | head -20
# (empty)

find /usr/src /opt /usr/local -name trtexec 2>/dev/null
# (empty)

python3 -c "import tensorrt as trt"
# ModuleNotFoundError: No module named 'tensorrt'

/home/user/Workspace/mergenvision/backend/.venv/bin/python -c "import tensorrt as trt; print(trt.__version__)"
# 10.16.1.11

test -f artifacts/models/insightface/buffalo_l/det_10g.onnx && echo ok
# ok

test -f artifacts/models/insightface/buffalo_l/w600k_r50.onnx && echo ok
# ok

ls -lh phase1/artifacts/engines/buffalo_l/ 2>/dev/null || echo "engines dir does not exist"
# engines dir does not exist

git status --short
# ?? phase1/
```

---

## 11. Risks and unknowns

1. **Tooling gap**: `trtexec` CLI is missing. The Python TensorRT API is available only in an old-project venv, which is not the clean production environment target.
2. **Dependency drift**: Reusing an old venv may introduce Python/package version constraints.
3. **Container alternative**: Pulling an `nvidia/tensorrt` image works on this host (Docker present), but adds image size and build-time complexity.
4. **No build log**: Without running a build, we cannot know whether the chosen optimization profiles load correctly, whether any operator is unsupported, or how long the build takes.

---

## 12. What is proven

- Source ONNX files are present and match prior SHA256 manifests.
- GPU is available and idle.
- The intended `trtexec` dynamic-shape command syntax aligns with NVIDIA's current documentation.
- The planned build strategy (detector static batch=1 640x640, recognizer dynamic min/opt/max) is consistent with the ONNX inspection report and the repo's TensorRT strategy document.

---

## 13. What is not proven yet

- TensorRT engine files can be built from these ONNX models.
- FP16 engines load and run without operator errors.
- Recognizer dynamic batch profile 1/8/32 is accepted by TensorRT.
- Batch invariance under TensorRT.
- ONNX vs TensorRT numerical equivalence.

---

## 14. Next recommended checkpoint

Choose one of the following, explicitly:

```text
APPROVED — PROCEED WITH PYTHON TENSORRT BUILDER USING EXISTING VENV
APPROVED — PULL NVIDIA TENSORRT DOCKER IMAGE AND BUILD WITH CONTAINERIZED trtexec
APPROVED — INSTALL TENSORRT SDK ON HOST AND RE-RUN THIS CHECKPOINT
```

If you choose the Python builder path, the script will use the existing `/home/user/Workspace/mergenvision/backend/.venv` or create a fresh local venv with `tensorrt==10.16.1.11` to build the engines without the `trtexec` binary.

If you choose the container path, the script will run `docker run --gpus all nvcr.io/nvidia/tensorrt` with the repo mounted, executing `trtexec` inside the container.

---

## 15. Turkish user-facing result summary

**Engine build bu oturumda BAŞLATILAMADI.**

Sebep: Bu makinede `trtexec` komutu bulunmuyor. `which trtexec` ve `/usr`, `/opt`, `/usr/local` altındaki aramalar boş döndü. Sistem Python’unda `tensorrt` modülü de yok.

Ancak eski `mergenvision` sanal ortamında (`/home/user/Workspace/mergenvision/backend/.venv`) TensorRT Python API kurulu: sürüm `10.16.1.11`. Yani model ağırlıkları ve ONNX dosyaları sağlam, GPU boşta, sadece `trtexec` CLI aracı eksik.

**Oluşan dosyalar:**

- `phase1/reports/BASELINE_TENSORRT_ENGINE_BUILD_REPORT.md`
- `phase1/reports/baseline_trt_engine_build.json`

**Oluşmayan dosyalar:**

- `phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan`
- `phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan`

**Neden batch=1 detector?**

- `det_10g.onnx` ONNX içinde batch boyutu zaten `1` olarak sabit; ONNX’i patchlemeden farklı batch ile build etmek güvenli değil.

**Neden dynamic profile recognizer?**

- `w600k_r50.onnx` girdisi `[None,3,112,112]`; bu yüzden `min=1 / opt=8 / max=32` profili planlandı.

**Neden henüz LFW/benchmark yapılmadı?**

- Bu checkpoint sadece TensorRT engine üretmek içindi; ONNX-TRT karşılaştırma, LFW ve benchmark sonraki adımlar.

**Sıradaki adımlar (onay bekleniyor):**

1. Mevcut venv’teki TensorRT Python API ile build etmeye devam et, veya
2. NVIDIA TensorRT Docker imajı çekip içinde `trtexec` ile build et, veya
3. Host’a TensorRT SDK kurup checkpoint’i yeniden çalıştır.

Lütfen yukarıdaki onay ifadelerinden birini gönderin.
