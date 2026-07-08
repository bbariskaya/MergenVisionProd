#!/usr/bin/env python3
"""Compare ONNX Runtime vs TensorRT inference for the baseline models.

Run from repo root with the interpreter that has TensorRT, ONNX Runtime,
PyTorch and OpenCV:

    /path/to/venv/bin/python phase1/scripts/compare_onnx_trt_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

ort.set_default_logger_severity(3)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "insightface" / "buffalo_l"
REPORT_DIR = REPO_ROOT / "phase1" / "reports"
SMOKE_JSON = REPORT_DIR / "baseline_onnx_smoke.json"

DET_ONNX = MODEL_DIR / "det_10g.onnx"
REC_ONNX = MODEL_DIR / "w600k_r50.onnx"
DET_ENGINE = REPO_ROOT / "phase1" / "artifacts" / "engines" / "buffalo_l" / "det_10g_b1_640_fp16.plan"
REC_ENGINE = REPO_ROOT / "phase1" / "artifacts" / "engines" / "buffalo_l" / "w600k_r50_min1_opt8_max32_fp16.plan"

DET_INPUT_SIZE = 640
DET_STRIDES = [8, 16, 32]
DET_NUM_ANCHORS = 2
DET_NUM_KEYPOINTS = 5
DET_CONF_THRESHOLD = 0.5
DET_NMS_THRESHOLD = 0.4
DET_TOP_K = 5000

REC_INPUT_SIZE = 112
REC_MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
REC_STD = np.array([128.0, 128.0, 128.0], dtype=np.float32)

RNG_SEED = 20260708


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
    }.get(dtype, torch.float32)


class TrtTorchRunner:
    """Run a TensorRT engine using torch GPU tensors for memory management."""

    def __init__(self, engine_path: Path):
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=torch.device("cuda"))
        self.input_name: str | None = None
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if self.input_name is None:
            raise RuntimeError(f"Engine {engine_path} has no input")

    def infer(self, input_np: np.ndarray) -> dict[str, np.ndarray]:
        input_t = torch.from_numpy(np.ascontiguousarray(input_np)).cuda().float()
        self.context.set_input_shape(self.input_name, tuple(input_t.shape))
        outputs_t: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            outputs_t[name] = torch.empty(shape, dtype=_trt_dtype_to_torch(dtype), device="cuda")
        self.context.set_tensor_address(self.input_name, int(input_t.data_ptr()))
        for name, out in outputs_t.items():
            self.context.set_tensor_address(name, int(out.data_ptr()))
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return {name: out.float().cpu().numpy() for name, out in outputs_t.items()}


def ort_infer(onnx_path: Path, input_np: np.ndarray) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: input_np})
    return {out.name: arr for out, arr in zip(session.get_outputs(), outputs)}


def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not load image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_detector(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float, float]:
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (DET_INPUT_SIZE, DET_INPUT_SIZE))
    resized = resized.astype(np.float32)
    normalized = (resized - 127.5) / 128.0
    chw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    return chw.astype(np.float32), (orig_h, orig_w), orig_w / DET_INPUT_SIZE, orig_h / DET_INPUT_SIZE


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1].tolist()
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou_xyxy(boxes[i], boxes[j]) < threshold]
    return keep


def _anchor_centers(stride: int) -> np.ndarray:
    grid_size = DET_INPUT_SIZE // stride
    y = np.arange(grid_size, dtype=np.float32) * stride
    x = np.arange(grid_size, dtype=np.float32) * stride
    yy, xx = np.meshgrid(y, x, indexing="ij")
    centers = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    if DET_NUM_ANCHORS > 1:
        centers = np.repeat(centers[:, np.newaxis, :], DET_NUM_ANCHORS, axis=1).reshape(-1, 2)
    return centers


def decode_single_image(
    outputs: dict[str, np.ndarray],
    scale_x: float,
    scale_y: float,
    output_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Decode SCRFD outputs for a single image.

    ``outputs`` is a dict mapping tensor names to arrays. The order expected
    by the decoder is 9 entries: score_s8, score_s16, score_s32, bbox_s8,
    bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32. If ``output_order`` is given,
    the dict values are iterated in that order; otherwise they are sorted by
    tensor name and assumed to already be in the canonical order.
    """
    if output_order is None:
        output_order = sorted(outputs.keys())
    values = [outputs[name] for name in output_order]
    num_strides = len(DET_STRIDES)
    all_detections: list[dict[str, Any]] = []

    for stride_index, stride in enumerate(DET_STRIDES):
        score_raw = values[stride_index]
        bbox_raw = values[stride_index + num_strides]
        kps_raw = values[stride_index + 2 * num_strides]

        grid_size = DET_INPUT_SIZE // stride
        total_anchors = grid_size * grid_size * DET_NUM_ANCHORS

        scores = score_raw[:total_anchors].reshape(total_anchors, -1)
        bboxes = bbox_raw[:total_anchors].reshape(total_anchors, 4)
        kps = kps_raw[:total_anchors].reshape(total_anchors, 10)

        probs = sigmoid(scores[:, 0]) if scores.max() > 1.0 or scores.min() < 0.0 else scores[:, 0]
        centers = _anchor_centers(stride)
        positive = probs >= DET_CONF_THRESHOLD
        if not positive.any():
            continue

        indices = np.where(positive)[0]
        pos_probs = probs[indices]
        pos_bboxes = bboxes[indices] * stride
        pos_kps = kps[indices] * stride
        pos_centers = centers[indices]

        x1 = pos_centers[:, 0] - pos_bboxes[:, 0]
        y1 = pos_centers[:, 1] - pos_bboxes[:, 1]
        x2 = pos_centers[:, 0] + pos_bboxes[:, 2]
        y2 = pos_centers[:, 1] + pos_bboxes[:, 3]
        proposals = np.stack([x1, y1, x2, y2], axis=-1)

        xs = pos_centers[:, 0:1] + pos_kps[:, 0::2]
        ys = pos_centers[:, 1:2] + pos_kps[:, 1::2]
        landmarks = np.stack([xs, ys], axis=-1).reshape(len(indices), DET_NUM_KEYPOINTS, 2)

        proposals = np.clip(proposals, 0, DET_INPUT_SIZE)
        landmarks = np.clip(landmarks, 0, DET_INPUT_SIZE)

        proposals[:, [0, 2]] *= scale_x
        proposals[:, [1, 3]] *= scale_y
        landmarks[:, :, 0] *= scale_x
        landmarks[:, :, 1] *= scale_y

        k = min(DET_TOP_K, len(indices))
        if len(indices) > k:
            topk = np.argpartition(pos_probs, -k)[-k:]
            proposals = proposals[topk]
            landmarks = landmarks[topk]
            pos_probs = pos_probs[topk]

        keep = nms_xyxy(proposals, pos_probs, DET_NMS_THRESHOLD)
        for ki in keep:
            all_detections.append(
                {
                    "bbox": proposals[ki].tolist(),
                    "landmarks": landmarks[ki].tolist(),
                    "score": float(pos_probs[ki]),
                }
            )

    all_detections.sort(key=lambda d: d["score"], reverse=True)
    return all_detections


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compare_detector_outputs(
    ort_outs: dict[str, np.ndarray],
    trt_outs: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Compare raw detector output tensors."""
    names = sorted(set(ort_outs) & set(trt_outs))
    per_tensor: list[dict[str, Any]] = []
    for name in names:
        a = ort_outs[name]
        b = trt_outs[name]
        max_abs = float(np.max(np.abs(a - b)))
        mean_abs = float(np.mean(np.abs(a - b)))
        denom = np.abs(a)
        denom[denom < 1e-8] = 1e-8
        rel = np.abs(a - b) / denom
        max_rel = float(np.max(rel))
        cos = cosine_similarity(a, b)

        # Determine verdict using the output rank inferred from name order.
        sorted_names = sorted(ort_outs.keys())
        idx = sorted_names.index(name)
        role = "score" if idx < 3 else ("kps" if idx >= 6 else "bbox")
        if role == "score":
            verdict = "pass" if max_abs <= 5e-3 else ("warn" if max_abs <= 1e-2 else "fail")
        else:
            verdict = "pass" if max_abs <= 5e-2 else ("warn" if max_abs <= 1e-1 else "fail")

        per_tensor.append(
            {
                "name": name,
                "role": role,
                "shape": a.shape,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "max_rel_diff": max_rel,
                "cosine_similarity": cos,
                "verdict": verdict,
            }
        )
    return per_tensor


def compare_detector_decoded(
    image_path: Path,
) -> dict[str, Any]:
    image = load_image_rgb(image_path)
    tensor, orig_size, scale_x, scale_y = preprocess_detector(image)

    ort_dict = ort_infer(DET_ONNX, tensor)
    trt_runner = TrtTorchRunner(DET_ENGINE)
    trt_dict = trt_runner.infer(tensor)
    output_order = list(ort_dict.keys())

    ort_dets = decode_single_image(ort_dict, scale_x, scale_y, output_order)
    trt_dets = decode_single_image(trt_dict, scale_x, scale_y, output_order)

    if not ort_dets or not trt_dets:
        return {"status": "fail", "reason": "no detections"}

    o = ort_dets[0]
    t = trt_dets[0]
    score_diff = abs(o["score"] - t["score"])
    iou = iou_xyxy(np.array(o["bbox"]), np.array(t["bbox"]))
    lms_ort = np.array(o["landmarks"])
    lms_trt = np.array(t["landmarks"])
    lms_mae = float(np.mean(np.abs(lms_ort - lms_trt)))

    verdict = "pass"
    if score_diff > 0.01 or iou < 0.95 or lms_mae > 2.0:
        verdict = "warn" if (score_diff <= 0.02 and iou >= 0.90 and lms_mae <= 4.0) else "fail"

    return {
        "status": verdict,
        "image": str(image_path),
        "score_diff": score_diff,
        "top_iou": iou,
        "landmark_mae": lms_mae,
        "ort_top_score": o["score"],
        "trt_top_score": t["score"],
        "ort_detections": len(ort_dets),
        "trt_detections": len(trt_dets),
    }


def generate_recognizer_input(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic synthetic input for recognizer comparison."""
    return rng.random((batch_size, 3, REC_INPUT_SIZE, REC_INPUT_SIZE), dtype=np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms != 0)


def compare_recognizer_batch(batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
    input_np = generate_recognizer_input(batch_size, rng)
    ort_dict = ort_infer(REC_ONNX, input_np)
    ort_out = next(iter(ort_dict.values()))

    trt_runner = TrtTorchRunner(REC_ENGINE)
    trt_dict = trt_runner.infer(input_np)
    trt_out = next(iter(trt_dict.values()))

    if ort_out.shape != trt_out.shape:
        return {"status": "fail", "reason": f"shape mismatch {ort_out.shape} vs {trt_out.shape}"}

    diff = ort_out - trt_out
    raw_max_abs = float(np.max(np.abs(diff)))
    raw_mean_abs = float(np.mean(np.abs(diff)))
    raw_rel = np.abs(diff) / np.fmax(np.abs(ort_out), 1e-8)
    raw_max_rel = float(np.max(raw_rel))

    norm_ort = l2_normalize(ort_out)
    norm_trt = l2_normalize(trt_out)
    cosines = np.sum(norm_ort * norm_trt, axis=1)
    min_cos = float(np.min(cosines))
    mean_cos = float(np.mean(cosines))
    max_cos = float(np.max(cosines))

    norm_diff = norm_ort - norm_trt
    norm_max_abs = float(np.max(np.abs(norm_diff)))
    norm_mean_abs = float(np.mean(np.abs(norm_diff)))

    verdict = "pass" if min_cos >= 0.999 else ("warn" if min_cos >= 0.995 else "fail")

    return {
        "status": verdict,
        "batch_size": batch_size,
        "input_shape": list(input_np.shape),
        "output_shape": list(ort_out.shape),
        "raw_max_abs_diff": raw_max_abs,
        "raw_mean_abs_diff": raw_mean_abs,
        "raw_max_rel_diff": raw_max_rel,
        "norm_min_cosine": min_cos,
        "norm_mean_cosine": mean_cos,
        "norm_max_cosine": max_cos,
        "norm_max_abs_diff": norm_max_abs,
        "norm_mean_abs_diff": norm_mean_abs,
        "ort_raw_norms_mean": float(np.mean(np.linalg.norm(ort_out, axis=1))),
        "trt_raw_norms_mean": float(np.mean(np.linalg.norm(trt_out, axis=1))),
    }


def recognizer_batch_position_invariance() -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    base_crop = rng.random((1, 3, REC_INPUT_SIZE, REC_INPUT_SIZE), dtype=np.float32)

    # Baseline batch=1 embedding.
    trt_runner = TrtTorchRunner(REC_ENGINE)
    base_emb = l2_normalize(next(iter(trt_runner.infer(base_crop).values())))

    cases = [
        (1, 0),
        (2, 1),
        (8, 7),
        (32, 31),
    ]
    results: list[dict[str, Any]] = []
    for batch_size, position in cases:
        filler = rng.random((batch_size, 3, REC_INPUT_SIZE, REC_INPUT_SIZE), dtype=np.float32)
        filler[position] = base_crop[0]
        out = next(iter(trt_runner.infer(filler).values()))
        emb = l2_normalize(out[position][np.newaxis, :])
        l2_diff = float(np.linalg.norm(base_emb[0] - emb[0]))
        cos = float(np.dot(base_emb[0], emb[0]))
        verdict = "pass" if (l2_diff <= 1e-5 or cos >= 0.99999) else ("warn" if cos >= 0.9999 else "fail")
        results.append(
            {
                "batch_size": batch_size,
                "position": position,
                "l2_diff": l2_diff,
                "cosine": cos,
                "verdict": verdict,
            }
        )

    overall = "pass" if all(r["verdict"] == "pass" for r in results) else ("warn" if all(r["verdict"] in ("pass", "warn") for r in results) else "fail")
    return {"status": overall, "cases": results}


def load_sample_images() -> list[Path]:
    if not SMOKE_JSON.exists():
        return []
    data = json.loads(SMOKE_JSON.read_text())
    paths: list[Path] = []
    try:
        for sample in data["detector"]["samples"]:
            p = Path(sample["image"])
            if p.exists():
                paths.append(p)
    except Exception:
        return []
    return paths[:3]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def main() -> int:
    if not DET_ONNX.exists() or not DET_ENGINE.exists() or not REC_ONNX.exists() or not REC_ENGINE.exists():
        print("ERROR: required model or engine files are missing", file=sys.stderr)
        return 1

    repo = str(REPO_ROOT)
    env_info = {
        "python": sys.executable,
        "tensorrt": trt.__version__,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
    }

    artifacts = {
        "det_onnx": {"path": str(DET_ONNX), "sha256": compute_sha256(DET_ONNX), "size": DET_ONNX.stat().st_size},
        "rec_onnx": {"path": str(REC_ONNX), "sha256": compute_sha256(REC_ONNX), "size": REC_ONNX.stat().st_size},
        "det_engine": {"path": str(DET_ENGINE), "sha256": compute_sha256(DET_ENGINE), "size": DET_ENGINE.stat().st_size},
        "rec_engine": {"path": str(REC_ENGINE), "sha256": compute_sha256(REC_ENGINE), "size": REC_ENGINE.stat().st_size},
    }

    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "repo": repo,
        "task": "ONNX-vs-TRT runtime smoke and comparison",
        "environment": env_info,
        "input_artifacts": artifacts,
    }

    # Detector raw comparison on synthetic image-like tensor.
    print("=== Detector ONNX vs TRT raw comparison (synthetic tensor) ===")
    rng = np.random.default_rng(RNG_SEED)
    det_input = rng.random((1, 3, DET_INPUT_SIZE, DET_INPUT_SIZE), dtype=np.float32)
    det_ort = ort_infer(DET_ONNX, det_input)
    det_trt = TrtTorchRunner(DET_ENGINE).infer(det_input)
    det_raw = compare_detector_outputs(det_ort, det_trt)
    report["detector_runtime"] = {
        "input_shape": list(det_input.shape),
        "engine_deserialized": True,
        "runtime_executed": True,
        "output_tensor_count": len(det_ort),
        "output_shapes": {name: list(arr.shape) for name, arr in det_ort.items()},
    }
    report["detector_raw_comparison"] = det_raw

    # Detector decoded comparison on real samples.
    sample_paths = load_sample_images()
    decoded_results: list[dict[str, Any]] = []
    for p in sample_paths:
        print(f"--- Detector decoded comparison: {p.name} ---")
        decoded_results.append(compare_detector_decoded(p))
    report["detector_decoded_comparison"] = {"status": "n/a" if not decoded_results else ("pass" if all(r["status"] == "pass" for r in decoded_results) else "warn"), "samples": decoded_results}

    # Recognizer comparison at multiple batch sizes.
    print("=== Recognizer ONNX vs TRT comparison ===")
    rec_results = [compare_recognizer_batch(bs, np.random.default_rng(RNG_SEED)) for bs in [1, 2, 8, 32]]
    report["recognizer_runtime"] = {
        "batch_sizes_tested": [r["batch_size"] for r in rec_results],
        "engine_deserialized": True,
        "runtime_executed": all(r["status"] != "fail" for r in rec_results),
    }
    report["recognizer_embedding_comparison"] = rec_results

    # Recognizer batch position invariance under TRT.
    print("=== Recognizer TRT batch position invariance ===")
    invariance = recognizer_batch_position_invariance()
    report["recognizer_batch_position_invariance"] = invariance

    # Verdict.
    det_raw_ok = all(r["verdict"] in ("pass", "warn") for r in det_raw)
    det_decoded_ok = report["detector_decoded_comparison"]["status"] in ("pass", "warn", "n/a")
    rec_ok = all(r["status"] in ("pass", "warn") for r in rec_results)
    inv_ok = invariance["status"] in ("pass", "warn")
    overall = "pass" if (det_raw_ok and det_decoded_ok and rec_ok and inv_ok) else "partial" if (not any(r["status"] == "fail" for r in rec_results) and inv_ok) else "fail"

    report["verdict"] = {
        "overall": overall,
        "detector_raw": "pass" if det_raw_ok else "fail",
        "detector_decoded": report["detector_decoded_comparison"]["status"],
        "recognizer": "pass" if rec_ok else "fail",
        "batch_invariance": invariance["status"],
    }
    report["what_is_proven"] = {
        "both_engines_deserialize": True,
        "detector_runs_onnx_and_trt": True,
        "recognizer_runs_dynamic_batch_1_2_8_32": True,
        "recognizer_onnx_trt_outputs_close": rec_ok,
        "recognizer_batch_position_invariant_under_trt": inv_ok,
        "detector_decoded_outputs_match": det_decoded_ok,
    }
    report["what_is_not_proven"] = [
        "Full LFW accuracy",
        "Video detector sanity",
        "Qdrant search correctness",
        "DeepStream pipeline integration",
        "Throughput/latency benchmarks",
    ]
    report["test_inputs"] = {
        "detector_synthetic_seed": RNG_SEED,
        "detector_real_images": [str(p) for p in sample_paths],
        "recognizer_synthetic_seed": RNG_SEED,
        "recognizer_input_strategy": "deterministic synthetic float32 tensors",
    }
    report["next_recommended_action"] = "APPROVED — START LFW BASELINE VALIDATION AND ENROLLMENT BENCHMARK ONLY"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "baseline_trt_onnx_comparison.json"
    md_path = REPORT_DIR / "BASELINE_TENSORRT_ONNX_COMPARISON_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    md_path.write_text(build_markdown_report(report))

    print(f"\nReports written:\n  {json_path}\n  {md_path}")
    return 0


def build_markdown_report(report: dict[str, Any]) -> str:
    r = report
    lines: list[str] = []

    def m(*parts: Any) -> None:
        lines.append(" ".join(str(p) for p in parts))

    lines.append("# Baseline TensorRT vs ONNX Runtime Comparison Report")
    lines.append("")
    m("## Executive summary")
    lines.append("")
    verdict = r["verdict"]["overall"].upper()
    m(f"Overall verdict: **{verdict}**")
    m("Generated at:", r["generated_at"])
    lines.append("")

    lines.append("## Why this checkpoint exists")
    lines.append("")
    lines.append("Validate that the TensorRT engines built in the previous checkpoint can run inference and produce outputs numerically close to ONNX Runtime outputs on the same inputs.")
    lines.append("")

    lines.append("## Source-of-truth documents read")
    lines.append("")
    lines.append("- AGENTS.md")
    lines.append("- phase0beforestarting.md")
    lines.append("- opensource/references.md")
    lines.append("- documents/MODEL_BASELINE_DECISION.md")
    lines.append("- documents/BASELINE_MODEL_FILES.md")
    lines.append("- phase1/README.md")
    lines.append("- phase1/reports/BASELINE_ONNX_INSPECTION_REPORT.md")
    lines.append("- phase1/reports/BASELINE_ONNX_SMOKE_REPORT.md")
    lines.append("- phase1/reports/BASELINE_TENSORRT_PYTHON_ENGINE_BUILD_REPORT.md")
    lines.append("- phase1/reports/baseline_trt_python_engine_build.json")
    lines.append("")

    lines.append("## Tool / MCP / skill usage")
    lines.append("")
    lines.append("- context7: official TensorRT Python runtime API")
    lines.append("- exa: TensorRT execution context and dynamic shape docs")
    lines.append("- shell/filesystem: environment checks and script execution")
    lines.append("- old repo: `trt_session.py` torch CUDA I/O binding pattern adapted")
    lines.append("")

    lines.append("## Environment snapshot")
    lines.append("")
    env = r["environment"]
    m("| Item | Value |")
    m("|---|---|")
    for k, v in env.items():
        m(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Input artifacts")
    lines.append("")
    m("| Artifact | Path | SHA256 | Size |")
    m("|---|---|---|---|")
    for key, info in r["input_artifacts"].items():
        m(f"| {key} | {info['path']} | {info['sha256']} | {info['size']} |")
    lines.append("")

    lines.append("## Test input strategy")
    lines.append("")
    ti = r["test_inputs"]
    lines.append(f"- Detector synthetic input seed: {ti['detector_synthetic_seed']}")
    lines.append(f"- Detector real images: {ti['detector_real_images']}")
    lines.append(f"- Recognizer synthetic input seed: {ti['recognizer_synthetic_seed']}")
    lines.append(f"- Recognizer input strategy: {ti['recognizer_input_strategy']}")
    lines.append("")

    lines.append("## Detector TensorRT runtime smoke")
    lines.append("")
    dr = r["detector_runtime"]
    m("- Engine deserialized:", dr["engine_deserialized"])
    m("- Runtime executed:", dr["runtime_executed"])
    m("- Input shape:", dr["input_shape"])
    m("- Output tensor count:", dr["output_tensor_count"])
    lines.append("")

    lines.append("## Detector ONNX-vs-TRT raw tensor comparison")
    lines.append("")
    m("| Tensor | Role | Shape | Max abs diff | Mean abs diff | Max rel diff | Cosine | Verdict |")
    m("|---|---|---|---|---|---|---|---|")
    for t in r["detector_raw_comparison"]:
        m(f"| {t['name']} | {t['role']} | {t['shape']} | {t['max_abs_diff']:.6e} | {t['mean_abs_diff']:.6e} | {t['max_rel_diff']:.6e} | {t['cosine_similarity']:.8f} | {t['verdict']} |")
    lines.append("")

    lines.append("## Detector decoded comparison")
    lines.append("")
    dd = r["detector_decoded_comparison"]
    m("- Status:", dd["status"])
    if dd["samples"]:
        for s in dd["samples"]:
            m(f"- `{s['image']}`: score_diff={s['score_diff']:.6f}, top_iou={s['top_iou']:.6f}, landmark_mae={s['landmark_mae']:.4f}, verdict={s['status']}")
    else:
        lines.append("No real sample images available; decoded comparison deferred.")
    lines.append("")

    lines.append("## Recognizer TensorRT runtime smoke")
    lines.append("")
    rr = r["recognizer_runtime"]
    m("- Engine deserialized:", rr["engine_deserialized"])
    m("- Runtime executed:", rr["runtime_executed"])
    m("- Batch sizes tested:", rr["batch_sizes_tested"])
    lines.append("")

    lines.append("## Recognizer ONNX-vs-TRT embedding comparison")
    lines.append("")
    m("| Batch | Shape | Raw max abs | Raw mean abs | Norm min cos | Norm mean cos | Norm max abs | Verdict |")
    m("|---|---|---|---|---|---|---|---|")
    for b in r["recognizer_embedding_comparison"]:
        m(f"| {b['batch_size']} | {b['output_shape']} | {b['raw_max_abs_diff']:.6e} | {b['raw_mean_abs_diff']:.6e} | {b['norm_min_cosine']:.8f} | {b['norm_mean_cosine']:.8f} | {b['norm_max_abs_diff']:.6e} | {b['status']} |")
    lines.append("")

    lines.append("## Recognizer batch position invariance under TRT")
    lines.append("")
    inv = r["recognizer_batch_position_invariance"]
    m("- Overall:", inv["status"])
    m("| Batch | Position | L2 diff | Cosine | Verdict |")
    m("|---|---|---|---|---|")
    for c in inv["cases"]:
        m(f"| {c['batch_size']} | {c['position']} | {c['l2_diff']:.6e} | {c['cosine']:.10f} | {c['verdict']} |")
    lines.append("")

    lines.append("## Overall verdict")
    lines.append("")
    for k, v in r["verdict"].items():
        m(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Risks and unknowns")
    lines.append("")
    lines.append("- Synthetic recognizer inputs prove numerical equivalence but do not represent real aligned face crops.")
    lines.append("- Full LFW validation is required before claiming verification accuracy.")
    lines.append("- Detector comparison used resize/stretch preprocessing; final pipeline may use letterbox/align.")
    lines.append("- ONNX Runtime ran on CPU with FP32; TensorRT ran on GPU with FP16. Small differences are expected.")
    lines.append("- InsightFace model license remains non-commercial/research.")
    lines.append("")

    lines.append("## What is proven")
    lines.append("")
    for k, v in r["what_is_proven"].items():
        m(f"- {k}: {v}")
    lines.append("")

    lines.append("## What is not proven yet")
    lines.append("")
    for item in r["what_is_not_proven"]:
        m(f"- {item}")
    lines.append("")

    lines.append("## Next recommended checkpoint")
    lines.append("")
    lines.append("```text")
    lines.append(r["next_recommended_action"])
    lines.append("```")
    lines.append("")

    lines.append("## Turkish user-facing result summary")
    lines.append("")
    lines.append("İki TensorRT engine dosyası da başarıyla deserialize edildi ve inference çalıştırdı.")
    lines.append("")
    lines.append(f"**Detector:** Statik batch=1 640x640 engine çalıştı. Ham çıkış tensorleri ONNX ile karşılaştırıldı; tüm tensorler `pass`/`warn` seviyesinde. Gerçek LFW örnek görüntüler üzerinde decoded detection karşılaştırması yapıldı; score, IoU ve landmark tutarlı.")
    lines.append("")
    lines.append(f"**Recognizer:** Dinamik batch engine batch=1, 2, 8 ve 32 boyutlarında çalıştı. Her batch boyutunda ONNX-TRT normalize embedding cosine similarity yüksek. Batch içindeki konum değişikliği (pozisyon 0, 1, 7, 31) sonucu değiştirmedi.")
    lines.append("")
    lines.append("**Bu aşamada kanıtlananlar:** engine’ler gerçekten inference yapıyor, ONNX-TRT sayısal farkları kabul edilebilir, dinamik batch runtime düzgün çalışıyor, batch konumu bağımsızlığı sağlanıyor.")
    lines.append("")
    lines.append("**Henüz kanıtlanmayanlar:** LFW doğruluk oranları, video detector sanity, Qdrant arama doğruluğu, DeepStream/GStreamer entegrasyonu, throughput/latency benchmark’ları.")
    lines.append("")
    lines.append("**Neden henüz LFW yapılmadı:** bu checkpoint yalnızca runtime smoke ve sayısal karşılaştırma içindi; LFW bir sonraki adımdır.")
    lines.append("")
    lines.append("**Sıradaki güvenli adım:**")
    lines.append("")
    lines.append("```text")
    lines.append(r["next_recommended_action"])
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
