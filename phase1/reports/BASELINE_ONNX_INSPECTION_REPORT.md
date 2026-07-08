# Baseline ONNX Inspection Report

## 1. Summary

First technical validation step for the InsightFace buffalo_l baseline stack. No TensorRT engine is built and no benchmark is run.

- Generated at: 2026-07-08T17:32:54.271411+00:00
- Models inspected: 2

## 2. Model files inspected

| File | Role | Local path | SHA256 | Size (bytes) |
|---|---|---|---|---|
| `det_10g.onnx` | detector | `/home/user/MergenVisionProd/artifacts/models/insightface/buffalo_l/det_10g.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | 16923827 |
| `w600k_r50.onnx` | recognizer | `/home/user/MergenVisionProd/artifacts/models/insightface/buffalo_l/w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` | 174383860 |

## 3. Detector: det_10g.onnx

- **Exists**: True
- **ONNX IR version**: 6
- **Producer**: pytorch 1.6
- **Initializer count**: 125
- **Node count**: 158

### Inputs

| Name | Shape | Dtype | Batch | Spatial |
|---|---|---|---|---|
| `input.1` | ['1', '3', '?', '?'] | float32 | static (1) | height=dynamic(?), width=dynamic(?) |

### Outputs

| Name | Shape | Dtype |
|---|---|---|
| `448` | ['12800', '1'] | float32 |
| `471` | ['3200', '1'] | float32 |
| `494` | ['800', '1'] | float32 |
| `451` | ['12800', '4'] | float32 |
| `474` | ['3200', '4'] | float32 |
| `497` | ['800', '4'] | float32 |
| `454` | ['12800', '10'] | float32 |
| `477` | ['3200', '10'] | float32 |
| `500` | ['800', '10'] | float32 |

- **Likely SCRFD outputs**: True
- **Landmarks likely present**: True
- **Output stride hints**: [8, 16, 32]
- **Inferred standard input size**: 640x640 (from SCRFD anchor counts)
- **Height/Width ONNX status**: H/W dims are not fixed in the ONNX graph; standard InsightFace input is 640x640.
- **TensorRT recommendation**: static_batch_1

## 4. Recognizer: w600k_r50.onnx

- **Exists**: True
- **ONNX IR version**: 6
- **Producer**: pytorch 1.9
- **Initializer count**: 237
- **Node count**: 130

### Inputs

| Name | Shape | Dtype | Batch | Spatial |
|---|---|---|---|---|
| `input.1` | ['None', '3', '112', '112'] | float32 | dynamic (None) | height=static(112), width=static(112) |

### Outputs

| Name | Shape | Dtype |
|---|---|---|
| `683` | ['None', '512'] | float32 |

- **Input looks like 3x112x112**: True
- **Output looks like 512-D embedding**: True
- **TensorRT recommendation**: dynamic_min_opt_max

## 5. Batch behavior verdict

- **Detector batch**: static
- **Recognizer batch**: dynamic
- **Detector safe batching**: Static batch=1. Do not feed batch>1 unless dynamic re-export is proven.
- **Recognizer safe batching**: Dynamic batch. Build min/opt/max profile and verify batch invariance.

Batch invariance must be verified later: same crop at batch indices 0,1,7,15,31,63 must produce identical embeddings/detections.

## 6. TensorRT engine strategy recommendation

**Detector**: Build detector TensorRT engine as batch=1 first. Do not claim batch>1 support. Later investigate dynamic export if needed.

**Recognizer**: Build recognizer TensorRT engine with min/opt/max profile, e.g. min=1, opt=32, max=64. Run batch invariance before trusting batch mode.

## 7. Risks and unknowns

- Pretrained weights remain non-commercial research only.
- Detector is static batch=1 in this ONNX. A separate dynamic-export effort is required if the final pipeline needs batched detection.
- DeepStream `nvinfer` custom parser for SCRFD outputs has not been written yet.
- ONNX shape inspection is static; numerical correctness and batch invariance require runtime tests.

## 8. Next recommended checkpoint

```text
APPROVED — START BASELINE ONNX RUNTIME BASELINE INFERENCE ONLY
```

After user approval, run ONNX Runtime inference on sample images to verify detector boxes/landmarks and recognizer 512-D embeddings.
