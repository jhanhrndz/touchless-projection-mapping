#!/usr/bin/env python3
"""
Herramienta de Benchmarking de Precisión por Etapas para el Motor de Visión
=============================================================================
Mide de forma aislada y con sincronización estricta GPU (torch.cuda.synchronize):
  1. Captura (lectura/decodificación del fotograma)
  2. MediaPipe (detección anatómica de manos)
  3. Inferencia (red neuronal en GPU / resta de fondo)
  4. Post-proceso (filtrado, componentes conectados, shader holográfico)
  5. Codificación (compresión MJPEG/JPEG)
  6. Total de extremo a extremo (FPS, media, p95, p99, VRAM, CPU)

Uso:
  python tools/benchmark.py [--source tests/clips/clip_a_hand_entry_borders.mp4] [--frames 60] [--output docs/benchmarks/baseline.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import psutil
import torch

# Rutas del entorno y proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision_engine import cuda_engine
from vision_engine.config import DEFAULT_TASK_PATH
from vision_engine.pipelines.yolo_pipeline import YoloArmDetector, FastSamArmDetector
from vision_engine.pipelines.rvm_background import RvmProcessor, BackgroundSubtractionProcessor
from vision_engine.hand_tracker import HandInteractionTracker
from vision_engine.gpu_renderer import GpuHologramRenderer


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    """Calcula media, p95, p99, min y max de una lista de métricas."""
    if not values:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.round(np.mean(arr), 2)),
        "p95": float(np.round(np.percentile(arr, 95), 2)),
        "p99": float(np.round(np.percentile(arr, 99), 2)),
        "min": float(np.round(np.min(arr), 2)),
        "max": float(np.round(np.max(arr), 2)),
    }


def run_benchmark_for_mode(
    mode_name: str,
    source_path: Path,
    num_frames: int = 60,
    warmup_frames: int = 10,
    device_str: str = "cuda"
) -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    process = psutil.Process()

    cap = cuda_engine.VideoFileSource(str(source_path))

    # Cargar modelos según el modo
    yolo_detector = None
    fastsam_detector = None
    rvm_processor = None
    bg_processor = None

    if "YOLO" in mode_name and cuda_engine.HAS_YOLO:
        yolo_detector = YoloArmDetector(device=device)

    if "FastSAM" in mode_name and cuda_engine.HAS_FASTSAM:
        fastsam_detector = FastSamArmDetector(device=device)

    if "RVM" in mode_name:
        rvm_processor = RvmProcessor(device=device)

    if "Resta de Fondo" in mode_name or "Fondo" in mode_name:
        bg_processor = BackgroundSubtractionProcessor()

    # MediaPipe
    task_p = DEFAULT_TASK_PATH
    hand_tracker = HandInteractionTracker(task_path=task_p if task_p.exists() else None)
    gpu_renderer = GpuHologramRenderer()

    # Métricas por etapa
    stage_times: dict[str, list[float]] = {
        "capture_ms": [],
        "mediapipe_ms": [],
        "inference_ms": [],
        "postprocess_ms": [],
        "encode_ms": [],
        "total_ms": [],
    }

    bg_frame = None
    rec = [None] * 4
    smooth_mask = None
    last_frame_ts = 0

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(0)

    total_evaluated = warmup_frames + num_frames

    for frame_idx in range(total_evaluated):
        is_warmup = frame_idx < warmup_frames

        # ETAPA 1: Captura
        t_cap_start = time.perf_counter()
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t_cap_end = time.perf_counter()
        cap_ms = (t_cap_end - t_cap_start) * 1000

        # ETAPA 2: MediaPipe (HandInteractionTracker v2)
        t_mp_start = time.perf_counter()
        hand_landmarks_list, _ = hand_tracker.process_frame(frame_rgb)
        t_mp_end = time.perf_counter()
        mp_ms = (t_mp_end - t_mp_start) * 1000

        # ETAPA 3: Inferencia
        if is_cuda:
            torch.cuda.synchronize(0)
        t_inf_start = time.perf_counter()

        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
        active_alpha = torch.zeros((1, 1, h, w), device=device)

        if "YOLO" in mode_name and yolo_detector is not None:
            active_alpha, _ = yolo_detector.predict_mask(frame_bgr, hand_landmarks_list, device=device)

        elif "FastSAM" in mode_name and fastsam_detector is not None:
            active_alpha, _ = fastsam_detector.predict_mask(frame_bgr, hand_landmarks_list, device=device)

        elif "RVM" in mode_name and rvm_processor is not None:
            variant = "resnet50" if "ResNet50" in mode_name else "mobilenetv3"
            active_alpha = rvm_processor.predict_alpha(
                frame_tensor,
                variant=variant,
                has_hands=(len(hand_landmarks_list) > 0),
                device=device,
            )

        elif ("Resta de Fondo" in mode_name or "Fondo" in mode_name) and bg_processor is not None:
            active_alpha, _ = bg_processor.compute_mask(frame_bgr, device=device)

        if is_cuda:
            torch.cuda.synchronize(0)
        t_inf_end = time.perf_counter()
        inf_ms = (t_inf_end - t_inf_start) * 1000

        # ETAPA 4: Post-proceso
        if is_cuda:
            torch.cuda.synchronize(0)
        t_post_start = time.perf_counter()

        # Aislamiento orgánico
        active_alpha = cuda_engine.isolate_arm_and_hand_mask(active_alpha, h, w, hand_landmarks_list, device)

        # Shader holográfico BGR en GPU
        _, output_bgr = gpu_renderer.render(frame_tensor, active_alpha, apply_guided_filter=False)

        if is_cuda:
            torch.cuda.synchronize(0)
        t_post_end = time.perf_counter()
        post_ms = (t_post_end - t_post_start) * 1000

        # ETAPA 5: Codificación
        t_enc_start = time.perf_counter()
        _, _ = cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        t_enc_end = time.perf_counter()
        enc_ms = (t_enc_end - t_enc_start) * 1000

        total_frame_ms = cap_ms + mp_ms + inf_ms + post_ms + enc_ms

        if not is_warmup:
            stage_times["capture_ms"].append(cap_ms)
            stage_times["mediapipe_ms"].append(mp_ms)
            stage_times["inference_ms"].append(inf_ms)
            stage_times["postprocess_ms"].append(post_ms)
            stage_times["encode_ms"].append(enc_ms)
            stage_times["total_ms"].append(total_frame_ms)

    cap.release()

    total_stats = calculate_percentiles(stage_times["total_ms"])
    avg_total_ms = total_stats["mean"] if total_stats["mean"] > 0 else 1.0
    effective_fps = round(1000.0 / avg_total_ms, 1)

    vram_used_mb = 0.0
    vram_peak_mb = 0.0
    if is_cuda:
        vram_used_mb = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1)
        vram_peak_mb = round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1)

    cpu_percent = round(process.cpu_percent(), 1)

    return {
        "mode": mode_name,
        "source": str(source_path.name),
        "frames_evaluated": len(stage_times["total_ms"]),
        "effective_fps": effective_fps,
        "latency_total": total_stats,
        "stages": {
            "capture": calculate_percentiles(stage_times["capture_ms"]),
            "mediapipe": calculate_percentiles(stage_times["mediapipe_ms"]),
            "inference": calculate_percentiles(stage_times["inference_ms"]),
            "postprocess": calculate_percentiles(stage_times["postprocess_ms"]),
            "encode": calculate_percentiles(stage_times["encode_ms"]),
        },
        "resources": {
            "vram_used_mb": vram_used_mb,
            "vram_peak_mb": vram_peak_mb,
            "cpu_percent": cpu_percent,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Banco de Pruebas y Benchmark de Rendimiento")
    parser.add_argument("--source", type=str, default="tests/clips/clip_a_hand_entry_borders.mp4",
                        help="Clip de prueba de entrada")
    parser.add_argument("--frames", type=int, default=60, help="Número de fotogramas a medir por modelo")
    parser.add_argument("--warmup", type=int, default=10, help="Fotogramas de calentamiento")
    parser.add_argument("--mode", type=str, default="all", help="Motor a evaluar ('all', 'yolo', 'fastsam', 'rvm_mobilenet', 'rvm_resnet', 'fondo')")
    parser.add_argument("--output", type=str, default="docs/benchmarks/baseline.json",
                        help="Ruta de guardado del reporte JSON")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"❌ Clip fuente no encontrado: {source_path}")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 80)
    print("  🚀 BANCO DE PRUEBAS DE VISIÓN - LÍNEA BASE (BENCHMARK SUITE)")
    print("=" * 80)
    print(f"  Dispositivo: {gpu_name}")
    print(f"  Clip fuente: {source_path.name}")
    print(f"  Fotogramas por motor: {args.frames} (calentamiento: {args.warmup})")
    print("=" * 80)

    all_modes = [
        "YOLO11-seg",
        "FastSAM Point-Prompt",
        "RVM MobileNetV3",
        "RVM ResNet50",
        "Resta de Fondo",
    ]

    req_mode = args.mode.lower()
    if req_mode == "all":
        modes = all_modes
    else:
        modes = [m for m in all_modes if any(k in m.lower() for k in req_mode.split(","))]
        if not modes:
            modes = all_modes

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": gpu_name,
            "cuda_available": torch.cuda.is_available(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        },
        "configuration": {
            "source_clip": str(source_path.name),
            "num_frames": args.frames,
            "warmup_frames": args.warmup,
        },
        "modes": {}
    }

    print(f"\n{'Motor':<22} | {'FPS':<6} | {'Total ms':<9} | {'Capt':<6} | {'MP':<6} | {'Inf':<6} | {'Post':<6} | {'Enc':<6} | {'VRAM (MB)':<10}")
    print("-" * 88)

    for m in modes:
        print(f"⏳ Evaluando: {m}...", flush=True)
        res = run_benchmark_for_mode(m, source_path, num_frames=args.frames, warmup_frames=args.warmup)
        results["modes"][m] = res
        tot = res["latency_total"]["mean"]
        st = res["stages"]
        vr = res["resources"]["vram_used_mb"]
        print(f"{m:<22} | {res['effective_fps']:<6.1f} | {tot:<9.1f} | {st['capture']['mean']:<6.1f} | {st['mediapipe']['mean']:<6.1f} | {st['inference']['mean']:<6.1f} | {st['postprocess']['mean']:<6.1f} | {st['encode']['mean']:<6.1f} | {vr:<10.1f}", flush=True)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 88, flush=True)
    print(f"\n[BENCHMARK] Resultados guardados exitosamente en: {out_path}", flush=True)


if __name__ == "__main__":
    main()
