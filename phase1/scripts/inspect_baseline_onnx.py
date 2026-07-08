#!/usr/bin/env python3
"""Inspect baseline ONNX models and write JSON + Markdown reports.

This script does NOT modify, patch, or build TensorRT engines.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "insightface" / "buffalo_l"
REPORT_DIR = REPO_ROOT / "phase1" / "reports"

DETECTOR_PATH = MODEL_DIR / "det_10g.onnx"
RECOGNIZER_PATH = MODEL_DIR / "w600k_r50.onnx"
JSON_REPORT = REPORT_DIR / "baseline_onnx_inspection.json"
MD_REPORT = REPORT_DIR / "BASELINE_ONNX_INSPECTION_REPORT.md"


TENSOR_TYPE_NAMES: dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.FLOAT16: "float16",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dim_to_str(dim: onnx.TensorShapeProto.Dimension) -> str:
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    if dim.HasField("dim_param"):
        return f"{dim.dim_param}"
    return "?"


def shape_list(shape: onnx.TensorShapeProto) -> list[str]:
    if shape is None:
        return []
    return [dim_to_str(d) for d in shape.dim]


def is_dim_dynamic(dim: onnx.TensorShapeProto.Dimension) -> bool:
    return dim.HasField("dim_param") or (not dim.HasField("dim_value"))


def batch_status(shape: onnx.TensorShapeProto) -> dict[str, Any]:
    if not shape.dim:
        return {"status": "unknown", "value": None}
    first = shape.dim[0]
    if first.HasField("dim_param"):
        return {"status": "dynamic", "value": first.dim_param}
    if first.HasField("dim_value"):
        return {"status": "static", "value": first.dim_value}
    return {"status": "unknown", "value": None}


def spatial_status(shape: onnx.TensorShapeProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dims = list(shape.dim)
    if len(dims) >= 3:
        for idx, name in ((2, "height"), (3, "width")):
            dim = dims[idx]
            if dim.HasField("dim_value"):
                result[name] = {"status": "static", "value": dim.dim_value}
            elif dim.HasField("dim_param"):
                result[name] = {"status": "dynamic", "value": dim.dim_param}
            else:
                result[name] = {"status": "unknown", "value": None}
    return result


def inspect_io(io_def: onnx.ValueInfoProto) -> dict[str, Any]:
    ttype = io_def.type.tensor_type
    shape = shape_list(ttype.shape)
    return {
        "name": io_def.name,
        "shape": shape,
        "dtype": TENSOR_TYPE_NAMES.get(ttype.elem_type, f"type_{ttype.elem_type}"),
        "batch_status": batch_status(ttype.shape),
        "spatial_status": spatial_status(ttype.shape),
    }


def analyze_detector_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic SCRFD detector analysis based on output channel patterns."""
    try:
        groups: dict[int, set[int]] = {}
        for out in outputs:
            if len(out["shape"]) != 2:
                continue
            n = int(out["shape"][0])
            c = int(out["shape"][1])
            groups.setdefault(n, set()).add(c)
    except Exception:
        return {
            "likely_scrfd": False,
            "landmarks_likely": False,
            "stride_hints": [],
            "inferred_input_size": None,
        }

    if len(groups) != 3:
        return {
            "likely_scrfd": False,
            "landmarks_likely": False,
            "stride_hints": sorted(groups.keys()) if groups else [],
            "inferred_input_size": None,
        }

    sorted_keys = sorted(groups.keys(), reverse=True)
    channels_per_group = [sorted(groups[k]) for k in sorted_keys]
    is_scrfd = all(set(chs) == {1, 4, 10} for chs in channels_per_group)
    has_landmarks = any(10 in chs for chs in channels_per_group)

    # Infer standard SCRFD strides [8, 16, 32] from anchor-count ratios.
    ratios = [k / sorted_keys[-1] for k in sorted_keys]
    stride_hints: list[int] = []
    inferred_size: int | None = None
    if is_scrfd and all(abs(r - expected) < 0.01 for r, expected in zip(ratios, [16.0, 4.0, 1.0])):
        stride_hints = [8, 16, 32]
        # N = 2 * (H / stride)^2. Largest key corresponds to stride 8.
        inferred_size = int(8 * (sorted_keys[0] / 2) ** 0.5)
    else:
        stride_hints = sorted_keys

    return {
        "likely_scrfd": is_scrfd,
        "landmarks_likely": has_landmarks,
        "stride_hints": stride_hints,
        "inferred_input_size": inferred_size,
    }


def inspect_model(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "file_exists": False, "role": role}

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    # Try shape inference to get output shapes when missing.
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        inferred = model

    inferred_graph = inferred.graph
    inferred_outputs = {v.name: v for v in inferred_graph.output}

    inputs = [inspect_io(inp) for inp in graph.input]
    outputs: list[dict[str, Any]] = []
    for out in graph.output:
        if out.name in inferred_outputs:
            outputs.append(inspect_io(inferred_outputs[out.name]))
        else:
            outputs.append(inspect_io(out))

    opset_imports = [
        {"domain": imp.domain, "version": imp.version}
        for imp in model.opset_import
    ]

    op_types = Counter(node.op_type for node in graph.node)

    record: dict[str, Any] = {
        "path": str(path),
        "file_exists": True,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
        "onnx_ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": getattr(model, "producer_version", ""),
        "opset_imports": opset_imports,
        "inputs": inputs,
        "outputs": outputs,
        "initializer_count": len(graph.initializer),
        "node_count": len(graph.node),
        "op_types_top20": dict(op_types.most_common(20)),
    }

    # Detector-specific heuristics
    if role == "detector":
        det_analysis = analyze_detector_outputs(outputs)
        record.update(det_analysis)

        batch = inputs[0]["batch_status"]["status"] if inputs else "unknown"
        if batch == "static" and inputs and inputs[0]["batch_status"].get("value") == 1:
            trt_rec = "static_batch_1"
        elif batch == "dynamic":
            trt_rec = "dynamic_possible"
        else:
            trt_rec = "needs_test"
        record["tensorrt_recommendation"] = trt_rec

    # Recognizer-specific heuristics
    elif role == "recognizer":
        batch = inputs[0]["batch_status"]["status"] if inputs else "unknown"
        expected_shape = ["", "3", "112", "112"]
        input_shape = inputs[0]["shape"] if inputs else []
        is_112 = len(input_shape) == 4 and input_shape[2:] == ["112", "112"]
        out_shape = outputs[0]["shape"] if outputs else []
        is_512 = len(out_shape) == 2 and out_shape[1] == "512"

        record["input_is_3x112x112"] = is_112
        record["output_is_512d"] = is_512

        if batch == "dynamic" and is_112:
            trt_rec = "dynamic_min_opt_max"
        elif batch == "static" and is_112:
            trt_rec = "static_batch_1"
        else:
            trt_rec = "needs_test"
        record["tensorrt_recommendation"] = trt_rec

        # Reflect dynamic batch in output shape/batch status.
        if batch == "dynamic" and outputs:
            outputs[0]["shape"][0] = inputs[0]["shape"][0]
            outputs[0]["batch_status"] = dict(inputs[0]["batch_status"])

    return record


def build_json_report(detector: dict[str, Any], recognizer: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_title": "Baseline ONNX Inspection Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "First technical validation step for the InsightFace buffalo_l baseline stack. "
            "No TensorRT engine is built and no benchmark is run."
        ),
        "models": [detector, recognizer],
        "batch_behavior_verdict": {
            "detector_batch_static_or_dynamic": detector.get("inputs", [{}])[0].get("batch_status", {}).get("status"),
            "recognizer_batch_static_or_dynamic": recognizer.get("inputs", [{}])[0].get("batch_status", {}).get("status"),
            "detector_safe_batch_note": "Static batch=1. Do not feed batch>1 unless dynamic re-export is proven.",
            "recognizer_safe_batch_note": "Dynamic batch. Build min/opt/max profile and verify batch invariance.",
        },
        "tensorrt_strategy": {
            "detector": tensorrt_strategy_text(detector, "detector"),
            "recognizer": tensorrt_strategy_text(recognizer, "recognizer"),
        },
    }


def tensorrt_strategy_text(model: dict[str, Any], role: str) -> str:
    rec = model.get("tensorrt_recommendation", "unknown")
    if role == "detector":
        if rec == "static_batch_1":
            return (
                "Build detector TensorRT engine as batch=1 first. "
                "Do not claim batch>1 support. Later investigate dynamic export if needed."
            )
        if rec == "dynamic_possible":
            return (
                "Dynamic batch appears possible. Build explicit batch profile after numerical validation."
            )
        return "Batch behavior unclear. Run batch invariance tests before deciding."

    if role == "recognizer":
        if rec == "dynamic_min_opt_max":
            return (
                "Build recognizer TensorRT engine with min/opt/max profile, e.g. min=1, opt=32, max=64. "
                "Run batch invariance before trusting batch mode."
            )
        if rec == "static_batch_1":
            return "Build recognizer engine as batch=1 and benchmark baseline."
        return "Batch behavior unclear. Run batch invariance tests before deciding."
    return "unknown"


def render_markdown(report: dict[str, Any], detector: dict[str, Any], recognizer: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Baseline ONNX Inspection Report")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(report["purpose"])
    lines.append("")
    lines.append(f"- Generated at: {report['generated_at']}")
    lines.append(f"- Models inspected: 2")
    lines.append("")

    lines.append("## 2. Model files inspected")
    lines.append("")
    lines.append("| File | Role | Local path | SHA256 | Size (bytes) |")
    lines.append("|---|---|---|---|---|")
    for m in [detector, recognizer]:
        lines.append(
            f"| `{Path(m['path']).name}` | {m['role']} | `{m['path']}` | `{m['sha256']}` | {m['file_size_bytes']} |"
        )
    lines.append("")

    lines.extend(render_model_section(detector, "detector"))
    lines.extend(render_model_section(recognizer, "recognizer"))

    lines.append("## 5. Batch behavior verdict")
    lines.append("")
    verdict = report["batch_behavior_verdict"]
    lines.append(f"- **Detector batch**: {verdict['detector_batch_static_or_dynamic']}")
    lines.append(f"- **Recognizer batch**: {verdict['recognizer_batch_static_or_dynamic']}")
    lines.append(f"- **Detector safe batching**: {verdict['detector_safe_batch_note']}")
    lines.append(f"- **Recognizer safe batching**: {verdict['recognizer_safe_batch_note']}")
    lines.append("")
    lines.append("Batch invariance must be verified later: same crop at batch indices 0,1,7,15,31,63 must produce identical embeddings/detections.")
    lines.append("")

    lines.append("## 6. TensorRT engine strategy recommendation")
    lines.append("")
    strategy = report["tensorrt_strategy"]
    lines.append(f"**Detector**: {strategy['detector']}")
    lines.append("")
    lines.append(f"**Recognizer**: {strategy['recognizer']}")
    lines.append("")

    lines.append("## 7. Risks and unknowns")
    lines.append("")
    lines.append("- Pretrained weights remain non-commercial research only.")
    lines.append("- Detector is static batch=1 in this ONNX. A separate dynamic-export effort is required if the final pipeline needs batched detection.")
    lines.append("- DeepStream `nvinfer` custom parser for SCRFD outputs has not been written yet.")
    lines.append("- ONNX shape inspection is static; numerical correctness and batch invariance require runtime tests.")
    lines.append("")

    lines.append("## 8. Next recommended checkpoint")
    lines.append("")
    lines.append("```text")
    lines.append("APPROVED — START BASELINE ONNX RUNTIME BASELINE INFERENCE ONLY")
    lines.append("```")
    lines.append("")
    lines.append("After user approval, run ONNX Runtime inference on sample images to verify detector boxes/landmarks and recognizer 512-D embeddings.")
    lines.append("")

    return "\n".join(lines)


def render_model_section(model: dict[str, Any], role: str) -> list[str]:
    title = "Detector" if role == "detector" else "Recognizer"
    section_num = "3" if role == "detector" else "4"
    lines: list[str] = []
    lines.append(f"## {section_num}. {title}: {Path(model['path']).name}")
    lines.append("")
    lines.append(f"- **Exists**: {model.get('file_exists')}")
    lines.append(f"- **ONNX IR version**: {model.get('onnx_ir_version')}")
    lines.append(f"- **Producer**: {model.get('producer_name')} {model.get('producer_version')}")
    lines.append(f"- **Initializer count**: {model.get('initializer_count')}")
    lines.append(f"- **Node count**: {model.get('node_count')}")
    lines.append("")

    lines.append("### Inputs")
    lines.append("")
    lines.append("| Name | Shape | Dtype | Batch | Spatial |")
    lines.append("|---|---|---|---|---|")
    for inp in model.get("inputs", []):
        batch = f"{inp['batch_status']['status']} ({inp['batch_status']['value']})"
        spatial = ", ".join(
            f"{k}={v['status']}({v['value']})" for k, v in inp.get("spatial_status", {}).items()
        )
        lines.append(f"| `{inp['name']}` | {inp['shape']} | {inp['dtype']} | {batch} | {spatial} |")
    lines.append("")

    lines.append("### Outputs")
    lines.append("")
    lines.append("| Name | Shape | Dtype |")
    lines.append("|---|---|---|")
    for out in model.get("outputs", []):
        lines.append(f"| `{out['name']}` | {out['shape']} | {out['dtype']} |")
    lines.append("")

    if role == "detector":
        lines.append(f"- **Likely SCRFD outputs**: {model.get('likely_scrfd')}")
        lines.append(f"- **Landmarks likely present**: {model.get('landmarks_likely')}")
        lines.append(f"- **Output stride hints**: {model.get('stride_hints')}")
        inferred = model.get("inferred_input_size")
        if inferred:
            lines.append(f"- **Inferred standard input size**: {inferred}x{inferred} (from SCRFD anchor counts)")
            lines.append(f"- **Height/Width ONNX status**: H/W dims are not fixed in the ONNX graph; standard InsightFace input is {inferred}x{inferred}.")

    if role == "recognizer":
        lines.append(f"- **Input looks like 3x112x112**: {model.get('input_is_3x112x112')}")
        lines.append(f"- **Output looks like 512-D embedding**: {model.get('output_is_512d')}")

    lines.append(f"- **TensorRT recommendation**: {model.get('tensorrt_recommendation')}")
    lines.append("")
    return lines


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    detector = inspect_model(DETECTOR_PATH, "detector")
    recognizer = inspect_model(RECOGNIZER_PATH, "recognizer")

    report = build_json_report(detector, recognizer)

    with open(JSON_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = render_markdown(report, detector, recognizer)
    with open(MD_REPORT, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"JSON report written to: {JSON_REPORT}")
    print(f"Markdown report written to: {MD_REPORT}")


if __name__ == "__main__":
    main()
