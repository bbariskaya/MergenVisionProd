# MergenVision Final Production Repository

This is the final MergenVision production repo.

> **Do not start backend code first.**

The current source-of-truth is the Phase 0 planning report:

- `phase0beforestarting.md` — source-code audit, GStreamer/DeepStream hotpath plan, model/TensorRT strategy, Qdrant design, API/database/worker plans, and 34-checkpoint roadmap.

Implementation order:

1. Governance and repo skeleton (this checkpoint).
2. Model source/license inventory (first baseline: InsightFace buffalo_l `det_10g.onnx` + `w600k_r50.onnx`).
3. Phase 1 Model Validation Lab / ModelLab harness (ONNX inspection, TensorRT engine build, batch invariance, LFW/WIDERFace benchmarks, Qdrant batch search, DeepStream smoke tests).
4. Only after the model validation gate passes: production backend code.

## Architecture target

Production video hotpath is **GStreamer + NVIDIA DeepStream**:

```text
video source
  -> NVDEC / GStreamer / DeepStream decode
  -> GPU/NVMM buffer
  -> nvstreammux
  -> TensorRT SCRFD detector
  -> nvtracker
  -> GPU crop + 5-point align
  -> TensorRT ArcFace recognizer
  -> GPU L2 normalization
  -> compact [N,512] embeddings + metadata to CPU
  -> Qdrant batch search
  -> PostgreSQL metadata / MinIO artifacts
```

Qdrant is the vector source of truth.
FAISS GPU may be used only as an optional benchmark accelerator.
PostgreSQL stores metadata.
MinIO stores images, videos, crops, and artifacts.

## Old repositories

The following are read-only references and lessons:

- `/home/user/Workspace/mergenvision`
- `/home/user/MergenVision`
- `/home/user/Demo/VideoFaceGpuLab`
- `/home/user/Workspace/MergenVisionCleanVersion`

Do not modify them. Do not copy old code blindly.

## Current next step

After this checkpoint:

```text
APPROVED — START CHECKPOINT 2 MODEL SOURCE AND LICENSE INVENTORY ONLY
```
