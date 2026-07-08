# backend/

This directory will contain the FastAPI backend, Alembic migrations, tests, and worker code.

**Backend implementation must not start until the model validation gate passes.**

No files under `backend/app/`, `backend/alembic/`, or `backend/scripts/` may be created until the appropriate checkpoint is approved.

Planned structure after Phase 1/2:

```text
backend/
  app/
    api/            # FastAPI routers (thin)
    application/    # business workflows/services
    domain/         # entities, enums, value objects
    repositories/   # SQLAlchemy CRUD
    infrastructure/ # adapters, db, vector store, storage
    ml/             # detector/recognizer/pipeline
    video/          # tracker, quality gate, decoder
    workers/        # DeepStream/GPU worker
  alembic/          # migrations
  tests/
  scripts/
```
