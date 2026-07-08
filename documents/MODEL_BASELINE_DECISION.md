# Model Baseline Decision

## 1. Decision summary

The first baseline stack to validate for MergenVision is the official InsightFace `buffalo_l` model pack, using only these two ONNX files:

| Role | File | Model | Source |
|---|---|---|---|
| Detector | `det_10g.onnx` | SCRFD-10GF with 5 keypoints | InsightFace `buffalo_l` model pack |
| Recognizer | `w600k_r50.onnx` | ArcFace ResNet50 trained on WebFace600K | InsightFace `buffalo_l` model pack |

## 2. Why buffalo_l det_10g + w600k_r50 is the first baseline

- This was the exact stack already used successfully in earlier MergenVision experiments.
- It has known-good ONNX files that work with the previous TensorRT pipeline.
- It reduces the number of unknowns while we validate the new DeepStream/GStreamer hotpath.
- The InsightFace `buffalo_l` pack is the official entry-level model distributed by the InsightFace project.

## 3. What this decision does NOT mean

This is **not** the final production model. This baseline is not final.
No stack is selected until the model validation gate passes.

This baseline must still pass:

- ONNX shape inspection
- ONNX Runtime baseline inference
- TensorRT engine build
- Batch invariance test where applicable
- LFW face verification
- Video detector sanity test
- Qdrant search correctness
- DeepStream nvinfer compatibility check
- Throughput and memory benchmarks

## 4. Later comparison candidates

After the baseline gate passes, we will compare against higher-accuracy or different-license stacks such as:

| Role | Candidate | Why compare |
|---|---|---|
| Detector | `SCRFD_34G_KPS` | Higher accuracy, more small-face recall |
| Detector | `SCRFD_10G_KPS` | Higher throughput sibling |
| Detector | `RetinaFace-R50` (OpenVINO impl.) | MIT-licensed commercial candidate |
| Recognizer | `ArcFace R100@Glint360K` (antelopev2) | Higher published LFW/CFP/AgeDB numbers |
| Recognizer | `AuraFace-v1` | Apache 2.0 commercial-friendly candidate |

## 5. License warning

InsightFace code is open-source, but pretrained model weights in the `buffalo_l` pack are distributed for **non-commercial research purposes** unless separately licensed by the authors.

```text
license_status: non-commercial / requires review
commercial_safe: false
```

For a commercial deployment, a separate licensing review would be required, or a switch to a verified commercial-friendly stack such as `RetinaFace-R50` (MIT) + `AuraFace-v1` (Apache 2.0).

## 6. Downloaded files

Local paths:

```text
artifacts/models/insightface/buffalo_l/det_10g.onnx
artifacts/models/insightface/buffalo_l/w600k_r50.onnx
```

Archive kept for traceability:

```text
artifacts/models/insightface/buffalo_l/source/buffalo_l.zip
```

## 7. Source URL

Primary official source verified:

```text
https://github.com/deepinsight/insightface/blob/master/python-package/README.md
```

The README lists `buffalo_l` as the default model pack and provides a download link.
Direct download URL used (from the official README table):

```text
https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing
```

The official `storage.insightface.ai` URL was attempted but returned DNS resolution failure; the Google Drive link above is the same official link documented in the InsightFace repository.

## 8. SHA256 checksums

| File | SHA256 |
|---|---|
| `det_10g.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |

## 9. File sizes

| File | Size | Type |
|---|---|---|
| `det_10g.onnx` | 17 MB | ONNX model |
| `w600k_r50.onnx` | 167 MB | ONNX model |

## 10. Current next step

Wait for user review of the downloaded baseline models.

Next checkpoint after approval:

```text
APPROVED — START CHECKPOINT 3 PHASE 1 MODEL VALIDATION LAB ONLY
```

No TensorRT engine build, benchmark, or backend code should run until the next checkpoint is explicitly approved.
