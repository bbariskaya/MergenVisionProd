# MergenVision Reference-First Implementation Policy

This file is mandatory for every agent working on `/home/user/MergenVision`.

Before implementing any feature, component, API, model pipeline, worker, database layer, frontend page, Docker setup, or test, the agent must first check the relevant references in this file.

Do not code only from memory.

The required workflow is:

```text
1. Read this references.md file.
2. Identify which reference links are relevant to the current task.
3. Inspect the implementation details from those references.
4. Check official docs when framework/runtime behavior matters.
5. Write a short reference mapping.
6. Only then implement.
7. Add tests.
8. Verify.
9. Report what references were checked.
```

For every major task, the agent must produce this before coding:

```text
REFERENCE_CHECK

Task:
- ...

Relevant references checked:
- ...

Implementation details found:
- ...

Patterns to follow:
- ...

Patterns rejected:
- ...

How this maps to MergenVision:
- ...

Files to implement:
- ...

Tests to write:
- ...
```

The agent must not say “I know how to implement this” and start coding without checking references.

The agent must always inspect reference implementation details for things like:

- face detection preprocessing
- SCRFD output decoding
- ArcFace embedding preprocessing
- face alignment
- ONNX Runtime provider/device handling
- FastAPI route structure
- SQLAlchemy model/migration structure
- PostgreSQL job queue / `SKIP LOCKED`
- Qdrant collection/search/upsert patterns
- MinIO bucket/object/presigned URL patterns
- Docker Compose / GPU worker patterns
- frontend dashboard/upload/table/status patterns
- Postman collection/test patterns

Do not copy-paste upstream source code blindly.

Use the references to understand implementation structure, lifecycle, naming, tests, and edge cases.

If implementation details are unclear, the agent must stop and say exactly what is unclear instead of guessing.

If the agent cannot verify a claim from references or local code, it must mark that claim as unverified.

Every final report must include:

```text
Reference proof:
- references checked:
- exact implementation details used:
- patterns adapted:
- patterns rejected:
- unverified parts:
```

If this reference proof is missing, the task is incomplete.

---

# Reference Links

## Face Recognition / Face Detection

https://github.com/deepinsight/insightface

https://insightface.ai/

https://github.com/deepinsight/insightface/tree/master/detection

https://github.com/deepinsight/insightface/tree/master/recognition

https://github.com/deepinsight/insightface/tree/master/model_zoo

https://github.com/deepinsight/insightface/tree/master/python-package

https://github.com/deepinsight/insightface/tree/master/examples

---

## Computer Vision Utilities / Detection Abstractions

https://github.com/roboflow/supervision

https://supervision.roboflow.com/

---

## Predictor / Model Runtime Pattern References

https://github.com/ultralytics/ultralytics

https://docs.ultralytics.com/

https://github.com/ultralytics/ultralytics/tree/main/ultralytics/engine

https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models

https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn

---

## Segment Anything / Future Segmentation Reference

https://github.com/facebookresearch/segment-anything

https://github.com/facebookresearch/segment-anything/tree/main/segment_anything

https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py

https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py

https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/transforms.py

---

## Runtime / Predictor Architecture References

https://github.com/PaddlePaddle/Paddle

https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html

https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/inference

---

## Face Recognition App / Prototype References

https://github.com/AarambhDevHub/multi-cam-face-tracker

---

## ONNX / Model Runtime

https://onnxruntime.ai/docs/

https://onnxruntime.ai/docs/api/python/api_summary.html

https://onnxruntime.ai/docs/execution-providers/

https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

https://onnxruntime.ai/docs/performance/

---

## Vector Database

https://qdrant.tech/documentation/

https://qdrant.tech/documentation/concepts/collections/

https://qdrant.tech/documentation/concepts/points/

https://qdrant.tech/documentation/concepts/search/

https://qdrant.tech/documentation/concepts/filtering/

https://qdrant.tech/documentation/guides/quantization/

https://python-client.qdrant.tech/

---

## Object Storage

https://min.io/docs/

https://min.io/docs/minio/linux/developers/python/API.html

https://min.io/docs/minio/linux/developers/python/minio-py.html

https://github.com/minio/minio-py

---

## Backend API

https://fastapi.tiangolo.com/

https://fastapi.tiangolo.com/tutorial/request-files/

https://fastapi.tiangolo.com/tutorial/bigger-applications/

https://fastapi.tiangolo.com/tutorial/dependencies/

https://fastapi.tiangolo.com/tutorial/handling-errors/

https://fastapi.tiangolo.com/reference/testclient/

---

## Database / ORM / Migrations

https://docs.sqlalchemy.org/

https://docs.sqlalchemy.org/en/20/orm/

https://docs.sqlalchemy.org/en/20/orm/quickstart.html

https://docs.sqlalchemy.org/en/20/orm/session_basics.html

https://alembic.sqlalchemy.org/

https://alembic.sqlalchemy.org/en/latest/tutorial.html

https://www.postgresql.org/docs/

https://www.postgresql.org/docs/current/sql-select.html

---

## Docker / GPU Runtime

https://docs.docker.com/compose/

https://docs.docker.com/compose/compose-file/

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html

---

## Frontend / UI

https://react.dev/

https://vite.dev/

https://tailwindcss.com/docs

https://reactrouter.com/

https://tanstack.com/query/latest/docs/framework/react/overview

https://vitest.dev/

https://testing-library.com/docs/react-testing-library/intro/

https://lucide.dev/icons/

---

## API Testing

https://learning.postman.com/docs/introduction/overview/

https://learning.postman.com/docs/collections/collections-overview/

https://learning.postman.com/docs/writing-scripts/test-scripts/

https://learning.postman.com/docs/collections/using-newman-cli/command-line-integration-with-newman/

https://github.com/postmanlabs/newman

---

## Optional Face Recognition / Similarity References

https://github.com/serengil/deepface

https://github.com/ageitgey/face_recognition

https://github.com/timesler/facenet-pytorch

---

## Optional Benchmark / Dataset References

https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

https://paperswithcode.com/task/face-recognition

https://paperswithcode.com/task/face-detection
