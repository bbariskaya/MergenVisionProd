# Baseline ONNX Runtime Smoke Report

## 1. Summary

ONNX Runtime smoke test for the InsightFace buffalo_l baseline stack. No TensorRT engine and no full benchmark.
- Generated at: 2026-07-08T17:50:25.816741+00:00

## 2. Detector smoke

| Image | Detections | Top score | Landmarks valid |
|---|---|---|---|
| `AJ_Cook_0001.jpg` | 1 | 0.809 | True |
| `AJ_Lamas_0001.jpg` | 1 | 0.831 | True |
| `Aaron_Eckhart_0001.jpg` | 1 | 0.854 | True |

## 3. Recognizer smoke

| Image | Embedding dim | Raw L2 norm | Norm. L2 norm | Same-crop cosine | Batch invariance |
|---|---|---|---|---|---|
| `AJ_Cook_0001.jpg` | 512 | 20.069223 | 1.000000 | 1.000000 | batch_1_max_l2_diff=0.00e+00, batch_2_max_l2_diff=0.00e+00, batch_8_max_l2_diff=0.00e+00 |
| `AJ_Lamas_0001.jpg` | 512 | 23.559057 | 1.000000 | 1.000000 | batch_1_max_l2_diff=0.00e+00, batch_2_max_l2_diff=0.00e+00, batch_8_max_l2_diff=0.00e+00 |
| `Aaron_Eckhart_0001.jpg` | 512 | 26.542633 | 1.000000 | 1.000000 | batch_1_max_l2_diff=0.00e+00, batch_2_max_l2_diff=0.00e+00, batch_8_max_l2_diff=0.00e+00 |

### Cross-face cosine matrix

| | AJ_Cook_0001.jpg | AJ_Lamas_0001.jpg | Aaron_Eckhart_0001.jpg |
|---|---|---|---|
| AJ_Cook_0001.jpg | 1.0000 | 0.0036 | 0.0639 |
| AJ_Lamas_0001.jpg | 0.0036 | 1.0000 | -0.0204 |
| Aaron_Eckhart_0001.jpg | 0.0639 | -0.0204 | 1.0000 |

## 4. Verdict

- Detector found faces in all samples: True
- Recognizer output shape is [N,512]: True
- Recognizer raw embeddings are NOT L2-normalized by the model (manual normalization required): True
- Manual L2 normalization produces unit vectors: True
- Recognizer batch invariance holds (max diff < 1e-4): True

## 5. Risks and next steps

- Detector output order assumed to be [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32].
- Smoke used simple square resize/stretch preprocessing, not the final letterbox crop pipeline.
- Recognizer raw output is not L2-normalized; production pipeline must apply L2-normalization before Qdrant.
- Full LFW benchmark, TensorRT engine build, and batch invariance under TensorRT are future checkpoints.

Next checkpoint:
```text
APPROVED — START BASELINE TENSORRT ENGINE BUILD ONLY
```
