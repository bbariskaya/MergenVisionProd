# MergenVision Open-Source Reference Index

Mandatory rule: every agent must read this file before implementation or any technical decision.

Workflow:

1. Read this file.
2. Identify relevant references for the current task.
3. Inspect upstream docs/source via DeepWiki, Context7, or Exa as needed.
4. Write a short reference mapping before coding.
5. Report checked references in every final response.

Only approved, license-verified sources may be used for models and engines.

## Reference table

| Category | Reference | URL/path | Why relevant | Must inspect before |
|---|---|---|---|---|
| DeepStream | NVIDIA DeepStream SDK docs | https://developer.nvidia.com/deepstream-sdk | Official DeepStream SDK documentation | GStreamer pipeline design |
| DeepStream | deepstream_python_apps | https://github.com/NVIDIA-AI-IOT/deepstream_python_apps | Python sample apps and probe/parser patterns | Python probe implementation |
| DeepStream | nvstreammux | (DeepStream plugin ref) | Batches frames for inference | nvstreammux config |
| DeepStream | nvinfer | (DeepStream plugin ref) | TensorRT inference plugin | detector/recognizer config |
| DeepStream | nvtracker / NvDCF | (DeepStream plugin ref) | Object tracking metadata | tracker integration |
| DeepStream | secondary GIE examples | deepstream_python_apps | Multi-stage GIE setup | recognizer as SGIE |
| DeepStream | zero-copy GPU buffer examples | deepstream_python_apps | memory:NVMM / pyds surface access | custom crop/align probe |
| DeepStream | custom parser examples | deepstream_python_apps | NvDsInferParseCustomFunc | SCRFD parser |
| TensorRT | dynamic shapes guide | https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dynamic-shapes | min/opt/max profiles | engine building |
| TensorRT | explicit batch | TensorRT docs | ONNX explicit batch semantics | engine building |
| TensorRT | optimization profiles | TensorRT docs | profile tuning | engine building |
| TensorRT | trtexec CLI | https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec | build engines from ONNX | engine building |
| TensorRT | FP16 / INT8 | TensorRT docs | precision modes | engine building / accuracy gate |
| Qdrant | query_points | https://qdrant.tech/documentation/concepts/search/ | single vector search | Qdrant adapter |
| Qdrant | query_batch_points | https://qdrant.tech/documentation/concepts/search/ | batch vector search | track-level recognition |
| Qdrant | HNSW tuning | https://qdrant.tech/documentation/concepts/indexing/#vector-index | index parameters | collection setup |
| Qdrant | quantization | https://qdrant.tech/documentation/guides/quantization/ | scalar/binary quantization | large-scale collection |
| Qdrant | payload indexing | https://qdrant.tech/documentation/concepts/indexing/#payload-index | payload index config | collection setup |
| Qdrant | 512-D cosine collection design | Qdrant docs | ArcFace vector storage | collection setup |
| Detection | InsightFace | https://github.com/deepinsight/insightface | SCRFD/ArcFace/buffalo_l source | model selection |
| Detection | SCRFD | https://github.com/deepinsight/insightface/tree/master/detection/scrfd | detector details | detector adapter/parser |
| Detection | SCRFD_10G_KPS | InsightFace model zoo | detector candidate | validation gate |
| Detection | SCRFD_34G_KPS | InsightFace model zoo | detector candidate | validation gate |
| Detection | RetinaFace | https://github.com/deepinsight/insightface/tree/master/detection/retinaface | detector reference | commercial fallback |
| Detection | RetinaFace-R50 / OpenVINO | https://github.com/openvinotoolkit/open_model_zoo | MIT-licensed detector | commercial fallback |
| Detection | DeepStream-compatible detectors | NVIDIA model zoo / NGC | alternative production detectors | DeepStream validation |
| Recognition | ArcFace | https://github.com/deepinsight/insightface/tree/master/recognition/arcface | recognizer source | recognizer adapter |
| Recognition | buffalo_l w600k_r50.onnx | InsightFace Python package / model zoo (official zip: https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing) | first baseline recognizer (R50@WebFace600K) | validation gate |
| Detection | buffalo_l det_10g.onnx | InsightFace Python package / model zoo (same official zip) | first baseline detector (SCRFD-10GF) | validation gate |
| Recognition | antelopev2 R100 / Glint360K | InsightFace model zoo | accuracy candidate (R100@Glint360K) | validation gate |
| Recognition | AuraFace-v1 | https://huggingface.co/fal/AuraFace-v1 | Apache 2.0 commercial-friendly fallback | commercial stack validation |
| Tracking | ByteTrack | https://github.com/ifzhang/ByteTrack | two-stage association | ByteTrack fallback design |
| Tracking | DeepSORT | https://github.com/nwojke/deep_sort | Kalman + ReID cascade | future improvement reference |
| Tracking | BoT-SORT | https://github.com/NirAharon/BoT-SORT | CMC + ReID | future improvement reference |
| Tracking | Norfair | https://github.com/tryolabs/norfair | hit counter + custom distance | design reference |
| Tracking | nvtracker / NvDCF | DeepStream docs | GPU tracker production target | tracker integration |
| Backend | FastAPI | https://fastapi.tiangolo.com/ | API framework | API routes |
| Backend | SQLAlchemy 2.0 async | https://docs.sqlalchemy.org/en/20/orm/ | async ORM | repositories |
| Backend | Alembic | https://alembic.sqlalchemy.org/ | migrations | database schema |
| Backend | MinIO Python client | https://min.io/docs/minio/linux/developers/python/API.html | object storage | storage adapter |
| Backend | PostgreSQL SKIP LOCKED | https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE | job queue primitive | worker queue |
| Local source-of-truth | Phase 0 report | /home/user/MergenVisionProd/phase0beforestarting.md | implementation roadmap | every task |
| Local source-of-truth | Architecture dir | /home/user/MergenVisionProd/architecture/ | architecture docs | every task |
| Reference repo | Clean current repo | /home/user/Workspace/mergenvision | Phase 1 working code | migration decisions |
| Reference repo | TensorRT reference | /home/user/MergenVision | engine/runtime references | TensorRT patterns |
| Reference repo | Video/tracker reference | /home/user/Demo/VideoFaceGpuLab | tracker/worker/queue references | video worker design |
| Reference repo | Clean architecture decisions | /home/user/Workspace/MergenVisionCleanVersion | locked architecture docs | architecture decisions |

## Verification workflow

For each reference actually inspected, record:

- exact URL/file path
- section or function inspected
- what pattern was learned
- how it maps to MergenVision
- license concern if any
