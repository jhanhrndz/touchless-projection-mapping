#!/usr/bin/env python3
"""
Grabador Interactivo de Clips de Prueba Reales
================================================
Guía al operador para grabar los clips reales etiquetados en `tests/clips/`:
  (a) clip_a_hand_entry_borders.mp4 : Mano entrando por cada borde
  (b) clip_b_hand_slow_fast.mp4     : Mano lenta y rápida (con pellizcos)
  (c) clip_c_robot_poster_static.mp4 : Robot / póster de persona (sin humano real)
  (d) clip_d_empty_wall_lighting.mp4: Pared vacía con cambios de luz
  (e) clip_e_two_hands.mp4          : Dos manos simultáneas
  (f) clip_f_torso_present.mp4       : Torso y cabeza visibles + brazo interactuando

Uso:
  python tools/record_clips.py [índice_cámara]
"""

import sys
import time
from pathlib import Path
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "clips"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 1280, 720
FPS = 30.0

SCENARIOS = [
    ("clip_a_hand_entry_borders.mp4", 10, "Mano entrando sucesivamente por cada borde (derecha, izquierda, abajo, arriba)"),
    ("clip_b_hand_slow_fast.mp4", 10, "Mano moviéndose lento (100 px/s) y luego rápido (900 px/s) con pellizcos"),
    ("clip_c_robot_poster_static.mp4", 8, "Escena con objeto/robot/poster humanoide SIN humano real"),
    ("clip_d_empty_wall_lighting.mp4", 8, "Pared vacía con luces cambiando / sombras del proyector"),
    ("clip_e_two_hands.mp4", 8, "Dos manos simultáneas entrando a la escena"),
    ("clip_f_torso_present.mp4", 10, "Torso y cabeza visibles mientras extiendes el brazo"),
]


def record_scenario(cap: cv2.VideoCapture, filename: str, duration_sec: int, description: str) -> None:
    filepath = OUTPUT_DIR / filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filepath), fourcc, FPS, (WIDTH, HEIGHT))

    print("\n" + "=" * 70)
    print(f"🎬 ESCENARIO: {filename}")
    print(f"📝 {description}")
    print(f"⏱️ Duración: {duration_sec} segundos")
    print("👉 Prepárate... la grabación iniciará en 3 segundos (Presiona ESC para cancelar)")
    print("=" * 70)

    for i in range(3, 0, -1):
        print(f"   [{i}]...")
        time.sleep(1.0)

    start_time = time.time()
    frames_recorded = 0
    print("🔴 GRABANDO AHORA...")

    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo fotograma de la cámara.")
            break

        out.write(frame)
        frames_recorded += 1

        # Mostrar preview
        preview = frame.copy()
        remaining = duration_sec - (time.time() - start_time)
        cv2.putText(preview, f"REC: {filename} ({remaining:.1f}s)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Grabador de Clips de Prueba", preview)
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("⚠️ Grabación cancelada por el usuario.")
            break

    out.release()
    print(f"✅ Guardado: {filepath.name} ({frames_recorded} cuadros, {filepath.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara #{cam_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Grabador de Clips de Prueba", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grabador de Clips de Prueba", 960, 540)

    try:
        for fname, dur, desc in SCENARIOS:
            record_scenario(cap, fname, dur, desc)
            print("\n¿Continuar al siguiente escenario? Presiona ENTER o escribe 'q' para salir: ", end="")
            ans = input().strip().lower()
            if ans == 'q':
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n🏁 Proceso de grabación finalizado.")


if __name__ == "__main__":
    main()
