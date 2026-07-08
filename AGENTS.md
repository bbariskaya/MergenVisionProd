# AGENTS.md — MergenVision ModelLab / GStreamer GPU Hotpath Rules

This repository exists to design, validate, and eventually implement the final clean MergenVision video face-recognition architecture.

The goal is not to build random code quickly.

The goal is to satisfy the project requirements with the most optimized GPU-first architecture possible:

```text
GStreamer / DeepStream video hotpath
  -> GPU decode
  -> GPU preprocessing
  -> TensorRT detector
  -> GPU tracker / metadata tracker where possible
  -> GPU crop / align
  -> TensorRT recognizer
  -> compact embedding / metadata crosses CPU boundary
  -> Qdrant batch search
  -> PostgreSQL metadata
  -> MinIO video/crop/artifact storage
```

CPU is allowed for I/O, orchestration, metadata, DB/Qdrant/MinIO calls, JSON/report generation, and explicit debug/test fallback only.

CPU is not allowed as the production video pixel hotpath.

---

## 1. Repository Purpose

This repo is for the new final MergenVision architecture.

Primary goals:

1. Research and choose the best detector and recognizer model stack.
2. Validate model license, accuracy, TensorRT compatibility, batch behavior, and DeepStream/GStreamer compatibility.
3. Design the final GStreamer / DeepStream GPU hotpath.
4. Keep Qdrant as the vector source of truth.
5. Avoid unreliable random online “batched” model files.
6. Build trusted TensorRT engines from validated ONNX models.
7. Prove performance with benchmarks before claiming optimization.
8. Keep every important architectural decision documented.

This repo must not become a messy experiment dump.

---

## 2. Hard Architecture Rules

### 2.1 GPU Hotpath Rule

The production video path must target:

```text
compressed video
  -> NVDEC / GStreamer / DeepStream decode
  -> GPU/NVMM buffer
  -> GPU preprocess / resize / normalize
  -> TensorRT face detector
  -> detector postprocess with minimal CPU boundary
  -> tracker
  -> GPU crop / align
  -> TensorRT face recognizer
  -> GPU L2 normalization
  -> compact embedding + metadata to CPU
  -> Qdrant batch search
  -> PostgreSQL/MinIO persistence
```

Allowed CPU work:

- file path handling
- API/job orchestration
- PostgreSQL metadata writes
- Qdrant requests
- MinIO upload/download
- compact embedding transfer
- track metadata
- JSON result generation
- benchmark report writing
- explicit debug/test fallback

Forbidden as production path:

- OpenCV CPU video decode
- ffmpeg CPU frame extraction
- decoded frame JPEG/PNG intermediates
- CPU resize/letterbox/normalize as main path
- CPU crop/align as main path
- silent GPU decode failure -> CPU fallback
- claiming “full GPU hotpath” without proving every stage

If GPU path fails, report the blocker. Do not silently downgrade.

---

## 3. GStreamer / DeepStream Decision

The preferred final video hotpath is GStreamer / DeepStream.

Agent must research and design around:

- `nvv4l2decoder`
- `uridecodebin`
- `nvstreammux`
- `nvinfer`
- `nvtracker`
- secondary GIE / ArcFace embedding stage
- custom SCRFD parser if required
- custom face alignment/crop if required
- zero-copy GPU/NVMM buffer handling
- compact metadata export to Python/backend

Even if the detector is static batch=1, GStreamer/DeepStream can still improve the system by:

- using NVDEC hardware decode
- reducing CPU frame copies
- removing OpenCV/ffmpeg CPU hotpath
- improving pipeline scheduling
- reducing Python per-frame overhead
- integrating TensorRT inference cleanly
- supporting multi-stream / multi-video throughput

However:

- static batch=1 detector limits maximum detector throughput
- dynamic batch detector is preferred for final high-throughput version
- recognizer should support dynamic batch
- ArcFace recognition batching is strongly preferred
- Qdrant query must be batched at track-level where possible

---

## 4. Qdrant Source of Truth Rule

Qdrant remains the vector store and identity-search source of truth.

FAISS GPU must not replace Qdrant as the main gallery.

FAISS GPU may be discussed only as:

- optional benchmark comparison
- optional future accelerator
- local experimental cache

But the production architecture must treat Qdrant as the official vector store.

Storage responsibilities:

| Data | PostgreSQL | Qdrant | MinIO |
|---|---|---|---|
| person metadata | yes | no | no |
| face identity metadata | yes | safe reference payload only | no |
| raw embeddings | no | yes | no |
| original images/videos | no | no | yes |
| crops/artifacts | no | no | yes |
| job/result metadata | yes | no | optional artifacts only |

Qdrant payload must be safe:

- `faceIdentityId`
- `sampleId`
- `personId` nullable
- `identityType`
- `isActive`
- `modelName`
- `modelVersion`
- `embeddingDimension`

Forbidden in Qdrant payload:

- raw national ID
- PII-heavy fields
- image bytes
- video bytes
- crop bytes

---

## 5. Model Selection Rules

Do not use random online “batched SCRFD” or “batched ArcFace” models without provenance.

Every model candidate must have:

- source URL/repo/model card
- license
- commercial usability status
- input shape
- output shape
- preprocessing
- ONNX availability
- TensorRT compatibility
- batch behavior
- benchmark evidence or validation plan
- risk notes

Preferred model sources to research:

Detector candidates:

- SCRFD_10G_KPS
- SCRFD_34G_KPS
- buffalo_l `det_10g.onnx`
- antelopev2 detector / RetinaFace
- RetinaFace-10GF
- serious DeepStream-compatible face detector alternatives
- newer credible detector candidates only if source and license are clear

Recognizer candidates:

- buffalo_l `w600k_r50.onnx`
- antelopev2 R100 / Glint360K
- InsightFace R50/R100 WebFace600K/Glint360K variants
- AuraFace-v1
- other commercial-friendly face embedding models with clear license
- newer credible recognizers only if source and license are clear

Do not assume “newer” means better.

Do not assume “large model” means production-ready.

Do not assume “batchable” means correct.

All candidates must pass validation.

---

## 6. Static vs Dynamic Batch Rule

Agent must understand and explain this correctly.

A static batch=1 engine:

```text
input: [1, 3, 320, 320]
```

can only accept batch 1.

It cannot safely accept:

```text
input: [64, 3, 320, 320]
```

unless the ONNX/model/engine was built for dynamic batch or a separate batch-64 engine exists.

A dynamic batch TensorRT engine requires dynamic ONNX input and optimization profiles:

```text
input: [-1, 3, 320, 320]

min = [1, 3, 320, 320]
opt = [8, 3, 320, 320]
max = [32, 3, 320, 320]
```

Recognizer example:

```text
input: [-1, 3, 112, 112]

min = [1, 3, 112, 112]
opt = [32, 3, 112, 112]
max = [64, 3, 112, 112]
```

Rules:

- recognizer dynamic batch is strongly preferred
- detector dynamic batch is preferred but not mandatory for first baseline
- detector batch postprocess must be batch-aware
- SCRFD postprocess must not assume batch=1 if engine is batched
- batch invariance test is mandatory
- do not patch ONNX batch dimension blindly
- if ONNX batch patching is attempted, validate numerically

---

## 7. Model Validation Gate

No model is accepted until it passes the validation gate.

Required validation:

1. License check
2. Commercial usability check
3. ONNX shape inspection
4. Preprocessing inspection
5. ONNXRuntime baseline inference
6. TensorRT engine build
7. ONNX vs TensorRT numerical comparison
8. Batch invariance test
9. Detector sanity on video frames
10. Detector small-face / blur / profile sanity test
11. LFW validation
12. CFP-FP / AgeDB if feasible
13. same-person vs different-person score distribution
14. threshold sweep
15. Qdrant search correctness test
16. Qdrant batch query benchmark
17. DeepStream/nvinfer compatibility check
18. end-to-end video benchmark
19. GPU memory benchmark
20. final `MODEL_SELECTION_REPORT.md`

Batch invariance test:

```text
same crop at batch index 0, 1, 7, 31, 63
batch=1 vs batch=64 comparison
embedding cosine difference within tolerance
identity result unchanged by batch position
detector output consistency if detector supports batch
```

If batch position changes output unexpectedly, the model/engine/postprocess fails.

---

## 8. Required Documents Before Backend Coding

Before building the final backend, agent must produce and review these decision files.

Minimum first decision file:

```text
projectchoices.md
```

Then, after review:

```text
MODEL_RESEARCH_REPORT.md
MODEL_CANDIDATES_MATRIX.md
GSTREAMER_DEEPSTREAM_QDRANT_ARCHITECTURE.md
TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md
MODEL_VALIDATION_PLAN.md
MODEL_SELECTION_REPORT.md
```

Backend implementation must not start until model and architecture decisions are validated.

---

## 9. Old Repo / Reference Repo Inspection Rule

Before major decisions, inspect old references if available.

Current important paths:

```text
/home/user/Workspace/mergenvision
/home/user/Workspace/mergenvision/projectultrareport.md
/home/user/Workspace/mergenvision/opensourcereferences/references.md
/home/user/MergenVision
/home/user/Demo/VideoFaceGpuLab
```

Use them as references and lessons learned only.

Do not blindly copy old code.

Known lessons from old/reference repos:

- old Python-only video pipeline can leave GPU performance unused
- detector/recognizer must be validated independently
- CPU fallback must not become product path
- docs/source API mismatch must be avoided
- worker queue strategy must be explicit
- trackId must not become faceIdentityId
- anonymous identities must be real `FaceIdentity` rows if persisted
- Qdrant must not contain unsafe PII payloads
- benchmark evidence is mandatory before performance claims

If old paths do not exist, report honestly and continue.

---

## 10. Compaction / New Session Reconstruction Rule

At the beginning of every new session, after context compaction, or when the agent is unsure, use `codebase-memory-mcp` if available.

The agent must reconstruct state before acting.

Required output:

```text
SESSION_RECONSTRUCTION:
- current repo:
- current task:
- current phase:
- source-of-truth docs read:
- old/reference repos checked:
- projectchoices.md status:
- model research status:
- model validation status:
- GStreamer/DeepStream decision:
- Qdrant decision:
- FAISS GPU decision:
- detector candidates:
- recognizer candidates:
- TensorRT batching status:
- DeepStream compatibility status:
- Qdrant collection/search status:
- benchmark status:
- open risks:
- blockers:
- what is proven:
- what is assumed:
- files likely relevant for this task:
```

If reconstruction fails, stop and ask the user.

Do not continue from memory alone.

---

## 11. Mandatory MCP / Tool Usage

For research, planning, architecture, model selection, and implementation tasks, use all relevant tools.

Required when available:

- `codebase-memory-mcp`
- `deepwiki`
- `exa` / web search
- `context7`
- shell/filesystem

Use official docs and source references whenever possible.

Use tool results to change decisions.

Do not fake tool usage.

Forbidden:

- `ruflo`
- `21st`
- `https://21st.dev/api/mcp`

Tool report required in final response:

```text
TOOLS_USED:
- codebase-memory-mcp:
  - used/skipped:
  - why:
  - what was checked:
  - what was learned:
- deepwiki:
  - used/skipped:
  - why:
  - what was checked:
  - what was learned:
- exa/web search:
  - used/skipped:
  - why:
  - what was checked:
  - what was learned:
- context7:
  - used/skipped:
  - why:
  - what was checked:
  - what was learned:
- shell/filesystem:
  - used/skipped:
  - why:
  - what was checked:
  - what was learned:
```

---

## 12. Mandatory Skills

Use and report these skills when relevant:

- brainstorming
- writing-plans
- executing-plans
- systematic-debugging
- verification-before-completion
- codebase-memory
- context7-mcp
- self-review/code-review

Skill report required:

```text
SKILLS_USED:
- brainstorming:
- writing-plans:
- executing-plans:
- systematic-debugging:
- verification-before-completion:
- codebase-memory:
- context7-mcp:
- self-review/code-review:
```

If a skill is skipped, explain why.

---

## 13. Reference-First Rule

Before coding or final recommendations, inspect references.

Required reference categories:

### GStreamer / DeepStream

- NVIDIA DeepStream official docs
- deepstream_python_apps
- `nvstreammux`
- `nvinfer`
- `nvtracker`
- secondary GIE examples
- zero-copy GPU buffer examples
- custom parser examples

### TensorRT

- dynamic shapes
- explicit batch
- optimization profiles
- FP16 / INT8
- `trtexec`
- profile min/opt/max
- engine shape limitations

### Qdrant

- batch search
- `query_batch_points` or current equivalent
- HNSW tuning
- quantization
- payload indexes
- filtering
- collection configuration for 512-D cosine embeddings

### Face Detection / Recognition

- InsightFace
- SCRFD
- RetinaFace
- ArcFace
- buffalo_l
- antelopev2
- AuraFace
- commercial-friendly face embedding alternatives

For every major decision, include:

```text
REFERENCE_VERIFICATION:
- changed/decided area:
- references checked:
- DeepWiki references:
- Exa/web sources:
- Context7 docs:
- official docs:
- model cards:
- files/classes/functions inspected:
- adopted patterns:
- adapted patterns:
- rejected patterns:
- risks:
- final verdict: pass / partial / fail
```

No checkpoint is complete without reference verification.

---

## 14. No Arbitrary Coding Rule

Do not invent core infrastructure patterns from scratch when good official or open-source references exist.

Before implementation:

1. Read local repo docs.
2. Read old/reference repo patterns.
3. Check DeepWiki.
4. Check Context7 official docs.
5. Use Exa/web search for current facts, model cards, licenses, and official docs.
6. Compare options.
7. Write the decision.
8. Only then code, if user explicitly approved implementation.

Do not code directly from assumptions.

---

## 15. Implementation Approval Rule

The agent must not implement code unless the user gives an explicit approval phrase.

Examples:

```text
APPROVED — WRITE PROJECTCHOICES.md ONLY
APPROVED — CREATE MODEL VALIDATION LAB ONLY
APPROVED — START TRUSTED MODEL DOWNLOAD + SHAPE INSPECTION ONLY
APPROVED — START GSTREAMER/DEEPSTREAM MODEL PROTOTYPE ONLY
APPROVED — START QDRANT BATCH SEARCH BENCHMARK ONLY
APPROVED — START TENSORRT ENGINE BUILDER ONLY
```

If approval is vague, ask for clarification.

Do not expand scope.

---

## 16. Git Safety Rule

Do not run:

```bash
git add
git commit
git push
git reset --hard
git clean
```

unless the user explicitly asks.

Always show `git status --short` before and after meaningful changes.

Do not hide dirty working tree state.

---

## 17. Docker / GPU Safety Rule

Do not kill unrelated GPU processes.

Do not kill VLLM or other user workloads.

Before GPU-heavy tests, run:

```bash
nvidia-smi
```

If multiple GPUs exist, prefer explicit GPU pinning:

```bash
CUDA_VISIBLE_DEVICES=1
```

or a user-approved GPU.

Container names, image names, volumes, and compose project names should be unique and lowercase.

Prefer prefix:

```text
mergenvision-
```

Avoid port/container/volume collisions with old repos.

---

## 18. Privacy / Security Rule

Do not store raw national IDs.

Do not store PII in Qdrant payload.

Do not store raw embeddings in PostgreSQL.

Do not store image/video bytes in PostgreSQL.

Use:

- PostgreSQL for metadata
- Qdrant for embeddings + safe payload
- MinIO for bytes/artifacts

If national ID integration is needed later:

- use `nationalIdHash`
- use `nationalIdMasked`
- use server-side pepper
- never expose hash publicly
- never put national ID in Qdrant

---

## 19. Identity / Tracking Rules

These rules apply when video identity logic is implemented.

```text
trackId != faceIdentityId
trackId != personId
```

Definitions:

- `trackId`: video-local temporal segment
- `faceIdentityId`: global recognition identity
- `personId`: business/person record
- one person can have many tracks
- one faceIdentity can appear in many videos/tracks
- multiple tracks can merge after recognition
- raw trackIds must be preserved for traceability

Predicted bbox rule:

- predicted bbox is annotation/debug only
- predicted bbox is not a real detection
- predicted bbox is never recognition candidate
- predicted bbox must have `recognitionEligible=false`

Recognition rule:

- recognition is per track, not every frame
- use top-K best observations
- Qdrant search should be batched
- final identity uses weighted vote + margin check
- conflicting votes become ambiguous
- unknown quality face becomes real anonymous `FaceIdentity` if persisted

---

## 20. Performance Claim Rule

Do not claim:

- “full GPU hotpath”
- “production ready”
- “best model”
- “fastest”
- “batch supported”
- “commercially safe”

unless evidence exists.

Evidence can include:

- code inspection
- official docs
- model card
- license text
- ONNX shape inspection
- TensorRT build log
- ONNX vs TensorRT comparison
- batch invariance test
- benchmark JSON
- LFW/CFP/AgeDB/IJB-C validation
- video detector sanity test
- Qdrant benchmark
- DeepStream smoke test

If evidence is missing, say:

```text
status: assumed
```

not:

```text
status: proven
```

---

## 21. Required Final Response Format

Every agent response after a task must end with:

```text
STATUS: pass / partial / fail

Kısa karar:
- task:
- repo:
- files changed:
- implementation changed:
- docs changed:
- tests run:
- benchmark run:
- blockers:

SESSION_RECONSTRUCTION:
- current phase:
- source-of-truth docs:
- old refs checked:
- what is proven:
- what is assumed:

TOOLS_USED:
- codebase-memory-mcp:
- deepwiki:
- exa/web search:
- context7:
- shell/filesystem:

SKILLS_USED:
- brainstorming:
- writing-plans:
- executing-plans:
- systematic-debugging:
- verification-before-completion:
- codebase-memory:
- context7-mcp:
- self-review/code-review:

REFERENCE_VERIFICATION:
- changed/decided area:
- references checked:
- adopted:
- adapted:
- rejected:
- risks:
- final verdict:

Open blockers:
- list

Next recommended action:
- exact approval phrase
```

Do not omit this format.

---

## 22. First Recommended Workflow

The recommended sequence for this repo is:

```text
1. projectchoices.md
2. MODEL_RESEARCH_REPORT.md
3. MODEL_CANDIDATES_MATRIX.md
4. GSTREAMER_DEEPSTREAM_QDRANT_ARCHITECTURE.md
5. TENSORRT_BATCHING_AND_ENGINE_STRATEGY.md
6. MODEL_VALIDATION_PLAN.md
7. trusted model download + license recording
8. ONNX shape inspection
9. ONNXRuntime baseline
10. TensorRT engine builder
11. batch invariance tests
12. LFW / CFP-FP / AgeDB / video sanity validation
13. Qdrant batch search benchmark
14. DeepStream/GStreamer prototype
15. final backend architecture
```

Do not skip ModelLab validation.

Do not start backend before model and hotpath decisions are proven.
