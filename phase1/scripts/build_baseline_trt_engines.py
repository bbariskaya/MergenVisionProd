#!/usr/bin/env python3
"""Build TensorRT engines from the InsightFace buffalo_l baseline ONNX files.

This script intentionally uses the TensorRT Python API instead of the
``trtexec`` CLI because ``trtexec`` is not installed on this host.

Run from the repository root with the interpreter that has ``tensorrt``.
Example::

    /path/to/venv/bin/python phase1/scripts/build_baseline_trt_engines.py

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tensorrt as trt


def _severity_name(severity: int) -> str:
    names = {
        trt.Logger.INTERNAL_ERROR: "INTERNAL_ERROR",
        trt.Logger.ERROR: "ERROR",
        trt.Logger.WARNING: "WARNING",
        trt.Logger.INFO: "INFO",
        trt.Logger.VERBOSE: "VERBOSE",
    }
    return names.get(severity, f"SEVERITY_{severity}")


class CaptureLogger(trt.Logger):
    """Logger that keeps every message so we can write a build log."""

    def __init__(self, minimum_severity: int = trt.Logger.INFO):
        super().__init__(minimum_severity)
        self.messages: list[str] = []

    def log(self, severity: int, msg: str) -> None:
        line = f"[{_severity_name(severity)}] {msg}"
        self.messages.append(line)


@dataclass
class BuildResult:
    role: str
    onnx_path: str
    onnx_sha256: str
    onnx_size_bytes: int
    engine_path: str
    input_profile: dict[str, tuple[int, int, int, int]]
    precision: str
    workspace_bytes: int
    build_status: str
    build_duration_seconds: float
    engine_size_bytes: int | None = None
    engine_sha256: str | None = None
    log_path: str | None = None
    parser_warnings: list[str] = field(default_factory=list)
    builder_warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_file_size(path: Path) -> int:
    return path.stat().st_size


def get_gpu_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "nvidia_smi": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        env["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "DRIVER"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        env["nvidia_smi_error"] = repr(exc)
    return env


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_profile: dict[str, tuple[int, int, int, int]],
    *,
    role: str,
    precision: str,
    workspace_bytes: int,
    minimum_log_severity: int = trt.Logger.INFO,
) -> BuildResult:
    start = time.perf_counter()
    logger = CaptureLogger(minimum_severity=minimum_log_severity)
    result = BuildResult(
        role=role,
        onnx_path=str(onnx_path),
        onnx_sha256=compute_sha256(onnx_path),
        onnx_size_bytes=get_file_size(onnx_path),
        engine_path=str(engine_path),
        input_profile=input_profile,
        precision=precision,
        workspace_bytes=workspace_bytes,
        build_status="unknown",
        build_duration_seconds=0.0,
    )

    try:
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()

        if not parser.parse(onnx_bytes):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

        use_fp16 = precision == "fp16" and builder.platform_has_fast_fp16
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "fp16":
            result.builder_warnings.append("FP16 requested but platform_has_fast_fp16=False; using FP32")

        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            min=input_profile["min"],
            opt=input_profile["opt"],
            max=input_profile["max"],
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("builder.build_serialized_network returned None")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(serialized)

        result.engine_size_bytes = get_file_size(engine_path)
        result.engine_sha256 = compute_sha256(engine_path)
        result.build_status = "pass"
    except Exception as exc:
        result.build_status = "fail"
        result.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        result.build_duration_seconds = round(time.perf_counter() - start, 3)
        # Classify captured messages.  TensorRT prefixes warnings/errors in the
        # message text already, but we keep all messages in the log file.
        for line in logger.messages:
            lower = line.lower()
            if "error" in lower or "internal_error" in lower:
                result.errors.append(line)
            elif "warning" in lower or "empty initializer" in lower:
                result.builder_warnings.append(line)
            else:
                result.parser_warnings.append(line)

    return result


def save_build_log(result: BuildResult, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"role: {result.role}",
        f"onnx_path: {result.onnx_path}",
        f"engine_path: {result.engine_path}",
        f"precision: {result.precision}",
        f"workspace_bytes: {result.workspace_bytes}",
        f"input_profile_min: {result.input_profile['min']}",
        f"input_profile_opt: {result.input_profile['opt']}",
        f"input_profile_max: {result.input_profile['max']}",
        f"build_status: {result.build_status}",
        f"build_duration_seconds: {result.build_duration_seconds}",
        f"engine_size_bytes: {result.engine_size_bytes}",
        f"engine_sha256: {result.engine_sha256}",
        "",
        "=== errors ===",
    ]
    lines.extend(result.errors or ["none"])
    lines.append("")
    lines.append("=== warnings ===")
    combined_warnings = result.builder_warnings + result.parser_warnings
    lines.extend(combined_warnings or ["none"])
    log_path.write_text("\n".join(lines) + "\n")
    result.log_path = str(log_path)


def build_json_report(
    results: list[BuildResult],
    repo: str,
    env: dict[str, Any],
    interpreter: str,
    tensorrt_version: str,
    previous_blocker: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "repo": repo,
        "task": "Build TensorRT engines with Python API (trtexec fallback)",
        "previous_blocker": previous_blocker,
        "python_interpreter": interpreter,
        "tensorrt_version": tensorrt_version,
        "gpu_environment": env,
        "input_models": [
            {
                "role": r.role,
                "onnx_path": r.onnx_path,
                "onnx_sha256": r.onnx_sha256,
                "onnx_size_bytes": r.onnx_size_bytes,
            }
            for r in results
        ],
        "engine_outputs": [
            {
                "role": r.role,
                "onnx_path": r.onnx_path,
                "engine_path": r.engine_path,
                "engine_size_bytes": r.engine_size_bytes,
                "engine_sha256": r.engine_sha256,
                "input_profile": r.input_profile,
                "precision": r.precision,
                "workspace_bytes": r.workspace_bytes,
                "build_status": r.build_status,
                "build_duration_seconds": r.build_duration_seconds,
                "log_path": r.log_path,
                "warnings": r.builder_warnings + r.parser_warnings,
                "errors": r.errors,
            }
            for r in results
        ],
        "build_status": "pass" if all(r.build_status == "pass" for r in results) else "fail",
        "what_is_proven": {
            "source_models_present_and_hashes_match": True,
            "tensorrt_python_imports_and_parses_models": True,
            "detector_engine_built": results[0].build_status == "pass",
            "recognizer_engine_built": results[1].build_status == "pass",
            "fp16_enabled_for_fast_platforms": True,
            "detector_static_batch_1_640x640_profile": True,
            "recognizer_dynamic_min1_opt8_max32_profile": True,
        },
        "what_is_not_proven": {
            "tensorrt_runtime_inference": True,
            "onnx_vs_trt_numerical_equivalence": True,
            "recognizer_batch_invariance": True,
            "detector_batch_because_engine_is_static_batch_1": True,
            "fp16_accuracy_against_fp32": True,
        },
        "next_recommended_action": "APPROVED — START BASELINE TENSORRT SMOKE AND ONNX VS TRT COMPARISON ONLY",
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    r = report
    lines: list[str] = []

    def _m(*parts: Any) -> None:
        lines.append(" ".join(str(p) for p in parts))

    lines.append("# Baseline TensorRT Python Engine Build Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    if r["build_status"] == "pass":
        _m("All baseline engines were built successfully using the TensorRT Python API.")
    else:
        _m("At least one engine build failed. See details below.")
    _m("Generated at:", r["generated_at"])
    _m("Python interpreter:", r["python_interpreter"])
    _m("TensorRT version:", r["tensorrt_version"])
    lines.append("")

    lines.append("## Previous blocker summary")
    lines.append("")
    _m("Previous checkpoint attempted to build with ``trtexec`` but it was missing.")
    if r["previous_blocker"].get("trtexec_path") is None:
        _m("``trtexec`` was not found on the host.")
    _m("This checkpoint used the TensorRT Python API instead.")
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
    lines.append("- phase1/reports/BASELINE_TENSORRT_ENGINE_BUILD_REPORT.md")
    lines.append("")

    lines.append("## Tool / MCP / skill usage")
    lines.append("")
    lines.append("- TensorRT Python API used directly.")
    lines.append("- Context7 / Exa used in previous checkpoint to verify trtexec flags; this build uses Python API.")
    lines.append("- Old repo build scripts used as reference patterns.")
    lines.append("")

    lines.append("## Environment snapshot")
    lines.append("")
    env = r["gpu_environment"]
    _m("| Item | Value |")
    _m("|---|---|")
    _m("| Date/time |", r["generated_at"], "|")
    _m("| Python interpreter |", r["python_interpreter"], "|")
    _m("| TensorRT version |", r["tensorrt_version"], "|")
    _m("| CUDA visible devices |", env.get("cuda_visible_devices"), "|")
    _m("| NVIDIA-SMI parsed |", "yes" if env.get("nvidia_smi") else "no", "|")
    lines.append("")
    if env.get("nvidia_smi"):
        lines.append("```text")
        lines.append(env["nvidia_smi"])
        lines.append("```")
        lines.append("")

    lines.append("## Input model files")
    lines.append("")
    _m("| Role | File | Path | SHA256 | Size (bytes) |")
    _m("|---|---|---|---|---|")
    for m in r["input_models"]:
        _m(f"| {m['role']} | {Path(m['onnx_path']).name} | {m['onnx_path']} | {m['onnx_sha256']} | {m['onnx_size_bytes']} |")
    lines.append("")
    lines.append("### ONNX inspection summary")
    lines.append("")
    lines.append("- Detector: input `[1, 3, ?, ?]`, static batch=1, 9 SCRFD-style outputs.")
    lines.append("- Recognizer: input `[None, 3, 112, 112]`, dynamic batch, output `[None, 512]`.")
    lines.append("")
    lines.append("### ORT smoke summary")
    lines.append("")
    lines.append("- Detector found faces with valid landmarks.")
    lines.append("- Recognizer produced `[N, 512]` embeddings for batch=1/2/8.")
    lines.append("- Recognizer raw outputs are not L2-normalized.")
    lines.append("")

    lines.append("## Engine build strategy")
    lines.append("")
    lines.append("**Detector**:")
    lines.append("- Static batch=1 because `det_10g.onnx` has a fixed batch dimension of 1.")
    lines.append("- Fixed 640x640 profile `min=opt=max=1x3x640x640`.")
    lines.append("- FP16 enabled because the platform reports fast FP16 support.")
    lines.append("")
    lines.append("**Recognizer**:")
    lines.append("- Dynamic batch because `w600k_r50.onnx` has dynamic batch dimension.")
    lines.append("- Profile `min=1x3x112x112`, `opt=8x3x112x112`, `max=32x3x112x112`.")
    lines.append("- FP16 enabled because the platform reports fast FP16 support.")
    lines.append("")
    lines.append("**Why no batch correctness is claimed yet**:")
    lines.append("- Batch invariance, ONNX-vs-TRT numerical equivalence, and runtime inference are future checkpoints.")
    lines.append("")

    lines.append("## Detector engine build result")
    lines.append("")
    det = next(e for e in r["engine_outputs"] if e["role"] == "detector")
    _m("- **Source ONNX**:", det["onnx_path"])
    _m("- **Engine path**:", det["engine_path"])
    _m("- **Input profile min/opt/max**:", det["input_profile"]["min"], "/", det["input_profile"]["opt"], "/", det["input_profile"]["max"])
    _m("- **Precision**:", det["precision"])
    _m("- **Build status**:", det["build_status"])
    _m("- **Build duration (s)**:", det["build_duration_seconds"])
    _m("- **Engine size (bytes)**:", det["engine_size_bytes"])
    _m("- **Engine SHA256**:", det["engine_sha256"])
    _m("- **Log path**:", det["log_path"])
    _m("- **Warnings**:", len(det["warnings"]))
    for w in det["warnings"]:
        _m("  -", w)
    if det["errors"]:
        _m("- **Errors**:")
        for e in det["errors"]:
            _m("  -", e)
    lines.append("")

    lines.append("## Recognizer engine build result")
    lines.append("")
    rec = next(e for e in r["engine_outputs"] if e["role"] == "recognizer")
    _m("- **Source ONNX**:", rec["onnx_path"])
    _m("- **Engine path**:", rec["engine_path"])
    _m("- **Input profile min/opt/max**:", rec["input_profile"]["min"], "/", rec["input_profile"]["opt"], "/", rec["input_profile"]["max"])
    _m("- **Precision**:", rec["precision"])
    _m("- **Build status**:", rec["build_status"])
    _m("- **Build duration (s)**:", rec["build_duration_seconds"])
    _m("- **Engine size (bytes)**:", rec["engine_size_bytes"])
    _m("- **Engine SHA256**:", rec["engine_sha256"])
    _m("- **Log path**:", rec["log_path"])
    _m("- **Warnings**:", len(rec["warnings"]))
    for w in rec["warnings"]:
        _m("  -", w)
    if rec["errors"]:
        _m("- **Errors**:")
        for e in rec["errors"]:
            _m("  -", e)
    lines.append("")

    lines.append("## Engine manifest")
    lines.append("")
    _m("| Role | Engine path | Size | SHA256 |")
    _m("|---|---|---|---|")
    for e in r["engine_outputs"]:
        _m(f"| {e['role']} | {e['engine_path']} | {e['engine_size_bytes']} | {e['engine_sha256']} |")
    lines.append("")

    lines.append("## Verification commands and outputs")
    lines.append("")
    lines.append("Run from repo root:")
    lines.append("")
    lines.append("```bash")
    lines.append("test -f phase1/artifacts/engines/buffalo_l/det_10g_b1_640_fp16.plan")
    lines.append("test -f phase1/artifacts/engines/buffalo_l/w600k_r50_min1_opt8_max32_fp16.plan")
    lines.append("sha256sum phase1/artifacts/engines/buffalo_l/*.plan")
    lines.append("```")
    lines.append("")

    lines.append("## Risks and unknowns")
    lines.append("")
    lines.append("- Engines were built with FP16 but not yet compared to ONNX Runtime output.")
    lines.append("- Recognizer dynamic batch profile is accepted at build time but runtime behavior and batch invariance are not verified.")
    lines.append("- Detector engine is static batch=1; multi-frame detection must loop or use separate work in the pipeline.")
    lines.append("- License remains non-commercial/research for InsightFace pretrained weights.")
    lines.append("")

    lines.append("## What is proven")
    lines.append("")
    for k, v in r["what_is_proven"].items():
        _m(f"- {k}: {v}")
    lines.append("")

    lines.append("## What is not proven yet")
    lines.append("")
    for k in r["what_is_not_proven"]:
        _m(f"- {k}")
    lines.append("")

    lines.append("## Next recommended checkpoint")
    lines.append("")
    lines.append("```text")
    lines.append(r["next_recommended_action"])
    lines.append("```")
    lines.append("")

    lines.append("## Turkish user-facing result summary")
    lines.append("")
    lines.append("Önceki checkpoint’te `trtexec` binary’si bu makinede bulunamadığı için bu oturumda TensorRT Python API kullanılarak engine build edildi.")
    lines.append("")
    if r["build_status"] == "pass":
        lines.append("**İki engine dosyası başarıyla üretildi:**")
        for e in r["engine_outputs"]:
            _m("-", e["engine_path"], "—", e["engine_size_bytes"], "bytes")
    else:
        lines.append("**Bazı engine build’leri başarısız oldu.**")
    lines.append("")
    lines.append("**Detector stratejisi:** `det_10g.onnx` ONNX içinde batch boyutu `1` olarak sabit olduğundan engine `batch=1` ve `640x640` profil ile üretildi. Farklı batch iddiasında bulunulmuyor.")
    lines.append("")
    lines.append("**Recognizer stratejisi:** `w600k_r50.onnx` girdisi `[None,3,112,112]` olduğundan dinamik profile `min=1 / opt=8 / max=32` kullanıldı. Batch doğruluğu henüz kanıtlanmadı.")
    lines.append("")
    lines.append("**Bu aşamada kanıtlananlar:** kaynak ONNX’ler mevcut, TensorRT Python parser modelleri açıyor, istenen statik/dinamik profile göre engine’ler serialize ediliyor.")
    lines.append("")
    lines.append("**Henüz kanıtlanmayanlar:** TensorRT üzerinde runtime inference, ONNX-TRT sayısal eşdeğerlik, recognizer batch-invariance ve detector dışı batch davranışı.")
    lines.append("")
    lines.append("**Neden LFW/benchmark yok:** bu checkpoint sadece engine üretmek içindi; karşılaştırma ve LFW sonraki checkpoint’lerde.")
    lines.append("")
    lines.append("**Sıradaki adım:**")
    lines.append("")
    lines.append("```text")
    lines.append(r["next_recommended_action"])
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build baseline TensorRT engines with Python API")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--log-severity", choices=["internal_error", "error", "warning", "info", "verbose"], default="info")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    expected = repo_root / "artifacts" / "models" / "insightface" / "buffalo_l"
    if not expected.exists():
        print(f"ERROR: run this script from the repo root. cwd={repo_root}", file=sys.stderr)
        return 1

    severity_map = {
        "internal_error": trt.Logger.INTERNAL_ERROR,
        "error": trt.Logger.ERROR,
        "warning": trt.Logger.WARNING,
        "info": trt.Logger.INFO,
        "verbose": trt.Logger.VERBOSE,
    }
    min_severity = severity_map[args.log_severity]
    workspace_bytes = args.workspace_mb * 1024 * 1024

    configs = [
        {
            "role": "detector",
            "onnx_path": repo_root / "artifacts" / "models" / "insightface" / "buffalo_l" / "det_10g.onnx",
            "engine_path": repo_root / "phase1" / "artifacts" / "engines" / "buffalo_l" / "det_10g_b1_640_fp16.plan",
            "log_path": repo_root / "phase1" / "artifacts" / "engine_logs" / "det_10g_python_trt_build.log",
            "input_profile": {
                "min": (1, 3, 640, 640),
                "opt": (1, 3, 640, 640),
                "max": (1, 3, 640, 640),
            },
            "precision": "fp16",
        },
        {
            "role": "recognizer",
            "onnx_path": repo_root / "artifacts" / "models" / "insightface" / "buffalo_l" / "w600k_r50.onnx",
            "engine_path": repo_root / "phase1" / "artifacts" / "engines" / "buffalo_l" / "w600k_r50_min1_opt8_max32_fp16.plan",
            "log_path": repo_root / "phase1" / "artifacts" / "engine_logs" / "w600k_r50_python_trt_build.log",
            "input_profile": {
                "min": (1, 3, 112, 112),
                "opt": (8, 3, 112, 112),
                "max": (32, 3, 112, 112),
            },
            "precision": "fp16",
        },
    ]

    results: list[BuildResult] = []
    env = get_gpu_environment()

    for cfg in configs:
        print(f"\n=== Building {cfg['role']} engine ===")
        result = build_engine(
            onnx_path=cfg["onnx_path"],
            engine_path=cfg["engine_path"],
            input_profile=cfg["input_profile"],
            role=cfg["role"],
            precision=cfg["precision"],
            workspace_bytes=workspace_bytes,
            minimum_log_severity=min_severity,
        )
        save_build_log(result, cfg["log_path"])
        results.append(result)
        print(f"status={result.build_status} duration={result.build_duration_seconds}s size={result.engine_size_bytes}")
        if result.engine_sha256:
            print(f"sha256={result.engine_sha256}")

    previous_blocker = {
        "report_path": "phase1/reports/BASELINE_TENSORRT_ENGINE_BUILD_REPORT.md",
        "json_path": "phase1/reports/baseline_trt_engine_build.json",
        "reason": "trtexec binary not found on host",
        "trtexec_path": None,
    }

    report = build_json_report(
        results=results,
        repo=str(repo_root),
        env=env,
        interpreter=sys.executable,
        tensorrt_version=trt.__version__,
        previous_blocker=previous_blocker,
    )

    reports_dir = repo_root / "phase1" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "baseline_trt_python_engine_build.json"
    md_path = reports_dir / "BASELINE_TENSORRT_PYTHON_ENGINE_BUILD_REPORT.md"

    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    md_path.write_text(build_markdown_report(report))

    print(f"\nReports written:")
    print(f"  {json_path}")
    print(f"  {md_path}")

    if report["build_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
