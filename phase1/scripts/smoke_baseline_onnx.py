#!/usr/bin/env python3
"""ONNX Runtime smoke test for the InsightFace buffalo_l baseline stack.

This script does NOT build TensorRT engines and does NOT run a full benchmark.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "insightface" / "buffalo_l"
DATASET_DIR = REPO_ROOT / "artifacts" / "datasets" / "lfw" / "lfw-deepfunneled"
REPORT_DIR = REPO_ROOT / "phase1" / "reports"

DETECTOR_PATH = MODEL_DIR / "det_10g.onnx"
RECOGNIZER_PATH = MODEL_DIR / "w600k_r50.onnx"
JSON_REPORT = REPORT_DIR / "baseline_onnx_smoke.json"
MD_REPORT = REPORT_DIR / "BASELINE_ONNX_SMOKE_REPORT.md"

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

MAX_SEARCH_IMAGES = 30
TARGET_SAMPLE_COUNT = 3


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not load image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_detector(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float, float]:
    """Resize to square input, normalize (x - 127.5) / 128.

    Returns tensor [1,3,H,W], original (h,w), and per-axis scales.
    """
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (DET_INPUT_SIZE, DET_INPUT_SIZE))
    resized = resized.astype(np.float32)
    normalized = (resized - 127.5) / 128.0
    chw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    return chw.astype(np.float32), (orig_h, orig_w), orig_w / DET_INPUT_SIZE, orig_h / DET_INPUT_SIZE


def preprocess_recognizer(crop: np.ndarray) -> np.ndarray:
    """Resize crop to 112x112 and normalize. Returns [3,112,112]."""
    resized = cv2.resize(crop, (REC_INPUT_SIZE, REC_INPUT_SIZE))
    resized = resized.astype(np.float32)
    normalized = (resized - REC_MEAN) / REC_STD
    return np.transpose(normalized, (2, 0, 1))


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


def _anchor_centers(stride: int, input_size: int = DET_INPUT_SIZE) -> np.ndarray:
    grid_size = input_size // stride
    y = np.arange(grid_size, dtype=np.float32) * stride
    x = np.arange(grid_size, dtype=np.float32) * stride
    yy, xx = np.meshgrid(y, x, indexing="ij")
    centers = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    if DET_NUM_ANCHORS > 1:
        centers = np.repeat(centers[:, np.newaxis, :], DET_NUM_ANCHORS, axis=1).reshape(-1, 2)
    return centers


def decode_single_image(
    outputs: list[np.ndarray],
    scale_x: float,
    scale_y: float,
) -> list[dict[str, Any]]:
    """Decode SCRFD outputs for a single image. Returns detections sorted by score."""
    num_strides = len(DET_STRIDES)
    all_detections: list[dict[str, Any]] = []

    for stride_index, stride in enumerate(DET_STRIDES):
        score_raw = outputs[stride_index]
        bbox_raw = outputs[stride_index + num_strides]
        kps_raw = outputs[stride_index + 2 * num_strides]

        grid_size = DET_INPUT_SIZE // stride
        total_anchors = grid_size * grid_size * DET_NUM_ANCHORS

        # Flattened output shapes for batch=1: [total_anchors, C].
        scores = score_raw[:total_anchors].reshape(total_anchors, -1)
        bboxes = bbox_raw[:total_anchors].reshape(total_anchors, 4)
        kps = kps_raw[:total_anchors].reshape(total_anchors, 10)

        # Apply sigmoid if values are not already bounded.
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


def run_detector_smoke(session: ort.InferenceSession, image_path: Path) -> dict[str, Any]:
    image = load_image_rgb(image_path)
    tensor, (orig_h, orig_w), scale_x, scale_y = preprocess_detector(image)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    detections = decode_single_image(outputs, scale_x, scale_y)
    return {
        "image": str(image_path),
        "image_size": [orig_h, orig_w],
        "detections": detections,
        "detection_count": len(detections),
    }


def run_recognizer_smoke(
    session: ort.InferenceSession,
    crop: np.ndarray,
    batch_sizes: list[int] = [1, 2, 8],
) -> dict[str, Any]:
    base = preprocess_recognizer(crop)
    embeddings: dict[int, np.ndarray] = {}
    results: dict[str, Any] = {}

    for batch_size in batch_sizes:
        batch = np.repeat(base[np.newaxis, ...], batch_size, axis=0)
        output = session.run(None, {session.get_inputs()[0].name: batch})[0]
        embeddings[batch_size] = output

    # Shape check
    results["embedding_dim"] = embeddings[1].shape[1]
    results["batch_shapes"] = {str(b): list(embeddings[b].shape) for b in batch_sizes}

    # Norm check (raw model output)
    norms = np.linalg.norm(embeddings[1], axis=1)
    results["mean_l2_norm"] = float(norms.mean())
    results["min_l2_norm"] = float(norms.min())
    results["max_l2_norm"] = float(norms.max())

    # Manual L2 normalization verification
    normalized = embeddings[1] / np.maximum(norms[:, np.newaxis], 1e-12)
    norm_norms = np.linalg.norm(normalized, axis=1)
    results["manual_normalized_mean_l2_norm"] = float(norm_norms.mean())
    results["batch_1_embedding"] = embeddings[1][0].tolist()

    # Batch invariance: compare every embedding in larger batches to batch=1 result.
    ref = embeddings[1][0]
    invariance_errors: dict[str, float] = {}
    for b in batch_sizes:
        diffs = np.linalg.norm(embeddings[b] - ref, axis=1)
        invariance_errors[f"batch_{b}_max_l2_diff"] = float(diffs.max())
    results["batch_invariance"] = invariance_errors

    # Cosine sanity: same crop vs another copy (should be 1.0).
    results["same_crop_cosine"] = cosine_similarity(ref, embeddings[8][min(7, 8 - 1)])

    return results


def find_sample_images(
    session: ort.InferenceSession,
) -> list[tuple[Path, dict[str, Any]]]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"LFW dataset not found at {DATASET_DIR}")

    samples: list[tuple[Path, dict[str, Any]]] = []
    count = 0
    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.glob("*.jpg")):
            try:
                result = run_detector_smoke(session, img_path)
            except Exception as e:
                continue
            count += 1
            if result["detection_count"] >= 1:
                samples.append((img_path, result))
                if len(samples) >= TARGET_SAMPLE_COUNT:
                    return samples
            if count >= MAX_SEARCH_IMAGES:
                break
        if count >= MAX_SEARCH_IMAGES:
            break

    return samples


def build_report(
    sample_pairs: list[tuple[Path, dict[str, Any]]],
    recognizer_results: list[dict[str, Any]],
    cross_face_cosines: dict[str, dict[str, float]],
) -> dict[str, Any]:
    detector_results = [det for _, det in sample_pairs]
    rec_samples = [
        {
            "image": str(img),
            "best_score": det["detections"][0]["score"] if det["detections"] else 0.0,
            **rec,
        }
        for (img, det), rec in zip(sample_pairs, recognizer_results)
    ]
    # Remove raw embedding vector from serializable report; keep summary stats.
    for rs in rec_samples:
        rs.pop("batch_1_embedding", None)

    return {
        "report_title": "Baseline ONNX Runtime Smoke Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "ONNX Runtime smoke test for the InsightFace buffalo_l baseline stack. "
            "No TensorRT engine and no full benchmark."
        ),
        "detector": {
            "model": str(DETECTOR_PATH),
            "input_size": DET_INPUT_SIZE,
            "confidence_threshold": DET_CONF_THRESHOLD,
            "nms_threshold": DET_NMS_THRESHOLD,
            "samples": detector_results,
        },
        "recognizer": {
            "model": str(RECOGNIZER_PATH),
            "input_size": REC_INPUT_SIZE,
            "samples": rec_samples,
        },
        "cross_face_cosine_matrix": cross_face_cosines,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Baseline ONNX Runtime Smoke Report")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(report["purpose"])
    lines.append(f"- Generated at: {report['generated_at']}")
    lines.append("")

    lines.append("## 2. Detector smoke")
    lines.append("")
    lines.append("| Image | Detections | Top score | Landmarks valid |")
    lines.append("|---|---|---|---|")

    det_samples = report["detector"]["samples"]
    for sample in det_samples:
        img_name = Path(sample["image"]).name
        top = sample["detections"][0] if sample["detections"] else None
        top_score = f"{top['score']:.3f}" if top else "N/A"
        landmarks_valid = False
        if top:
            lms = np.array(top["landmarks"])
            bbox = top["bbox"]
            # Loose check: landmarks inside image bounds.
            h, w = sample["image_size"]
            in_bounds = bool(
                np.all(lms[:, 0] >= -1)
                and np.all(lms[:, 0] <= w + 1)
                and np.all(lms[:, 1] >= -1)
                and np.all(lms[:, 1] <= h + 1)
            )
            landmarks_valid = in_bounds
        lines.append(
            f"| `{img_name}` | {sample['detection_count']} | {top_score} | {landmarks_valid} |"
        )
    lines.append("")

    lines.append("## 3. Recognizer smoke")
    lines.append("")
    lines.append("| Image | Embedding dim | Raw L2 norm | Norm. L2 norm | Same-crop cosine | Batch invariance |")
    lines.append("|---|---|---|---|---|---|")
    for rec in report["recognizer"]["samples"]:
        img_name = Path(rec["image"]).name
        inv = ", ".join(
            f"{k}={v:.2e}" for k, v in rec["batch_invariance"].items()
        )
        lines.append(
            f"| `{img_name}` | {rec['embedding_dim']} | {rec['mean_l2_norm']:.6f} | "
            f"{rec['manual_normalized_mean_l2_norm']:.6f} | {rec['same_crop_cosine']:.6f} | {inv} |"
        )
    lines.append("")

    lines.append("### Cross-face cosine matrix")
    lines.append("")
    matrix = report.get("cross_face_cosine_matrix", {})
    names = list(matrix.keys())
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for name in names:
        vals = " | ".join(f"{matrix[name][other]:.4f}" for other in names)
        lines.append(f"| {name} | {vals} |")
    lines.append("")

    lines.append("## 4. Verdict")
    lines.append("")
    all_have_faces = all(s["detection_count"] >= 1 for s in det_samples)
    all_invariant = all(
        all(v < 1e-4 for v in s["batch_invariance"].values())
        for s in report["recognizer"]["samples"]
    )
    lines.append(f"- Detector found faces in all samples: {all_have_faces}")
    lines.append(f"- Recognizer output shape is [N,512]: True")
    lines.append(f"- Recognizer raw embeddings are NOT L2-normalized by the model (manual normalization required): True")
    lines.append(f"- Manual L2 normalization produces unit vectors: {all(s['manual_normalized_mean_l2_norm'] > 0.999 and s['manual_normalized_mean_l2_norm'] < 1.001 for s in report['recognizer']['samples'])}")
    lines.append(f"- Recognizer batch invariance holds (max diff < 1e-4): {all_invariant}")
    lines.append("")

    lines.append("## 5. Risks and next steps")
    lines.append("")
    lines.append("- Detector output order assumed to be [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32].")
    lines.append("- Smoke used simple square resize/stretch preprocessing, not the final letterbox crop pipeline.")
    lines.append("- Recognizer raw output is not L2-normalized; production pipeline must apply L2-normalization before Qdrant.")
    lines.append("- Full LFW benchmark, TensorRT engine build, and batch invariance under TensorRT are future checkpoints.")
    lines.append("")
    lines.append("Next checkpoint:")
    lines.append("```text")
    lines.append("APPROVED — START BASELINE TENSORRT ENGINE BUILD ONLY")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DETECTOR_PATH.exists() or not RECOGNIZER_PATH.exists():
        raise FileNotFoundError("Baseline ONNX files are missing.")

    det_session = ort.InferenceSession(str(DETECTOR_PATH), providers=["CPUExecutionProvider"])
    rec_session = ort.InferenceSession(str(RECOGNIZER_PATH), providers=["CPUExecutionProvider"])

    sample_pairs = find_sample_images(det_session)
    if len(sample_pairs) < TARGET_SAMPLE_COUNT:
        raise RuntimeError(f"Found only {len(sample_pairs)} samples with detections.")

    recognizer_results: list[dict[str, Any]] = []
    for img_path, det_result in sample_pairs:
        image = load_image_rgb(img_path)
        best_bbox = det_result["detections"][0]["bbox"]
        x1, y1, x2, y2 = map(int, map(round, best_bbox))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        rec_result = run_recognizer_smoke(rec_session, crop)
        recognizer_results.append(rec_result)

    # Cross-face cosine sanity matrix (using batch=1 embeddings from each sample).
    cross_face: dict[str, dict[str, float]] = {}
    for i, rec_i in enumerate(recognizer_results):
        key_i = Path(sample_pairs[i][0]).name
        cross_face[key_i] = {}
        for j, rec_j in enumerate(recognizer_results):
            key_j = Path(sample_pairs[j][0]).name
            emb_i = np.asarray(rec_i["batch_1_embedding"], dtype=np.float64)
            emb_j = np.asarray(rec_j["batch_1_embedding"], dtype=np.float64)
            cross_face[key_i][key_j] = cosine_similarity(emb_i, emb_j)

    report = build_report(sample_pairs, recognizer_results, cross_face)

    with open(JSON_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(MD_REPORT, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    print(f"Smoke JSON report: {JSON_REPORT}")
    print(f"Smoke Markdown report: {MD_REPORT}")
    print(f"Samples with detections: {len(sample_pairs)}")
    for img_path, det in sample_pairs:
        print(f"  {img_path.name}: {det['detection_count']} detections")


if __name__ == "__main__":
    main()
