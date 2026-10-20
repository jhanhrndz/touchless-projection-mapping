#!/usr/bin/env python3
"""
Generador de Clips de Prueba Sintéticos para Banco de Pruebas de Visión
========================================================================
Genera secuencias de video MP4 a 1280x720 @ 30 FPS en `tests/clips/` para los
escenarios de evaluación obligatorios:
  (a) Mano entrando por cada borde
  (b) Mano lenta y rápida
  (c) Escena con robot/figura humanoide estática (sin humano real)
  (d) Pared vacía con cambios de iluminación
  (e) Dos manos simultáneas
  (f) Torso presente junto con brazo extendido
"""

import math
from pathlib import Path
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "clips"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIDTH = 1280
HEIGHT = 720
FPS = 30.0


def draw_realistic_hand(
    frame: np.ndarray,
    wrist_pt: tuple[int, int],
    palm_pt: tuple[int, int],
    finger_spread: float = 1.0,
    pinch: bool = False,
    skin_color: tuple[int, int, int] = (165, 195, 225),  # BGR piel cálida
    clothing_color: tuple[int, int, int] = (60, 60, 180),  # Manga oscura
) -> None:
    """Dibuja un brazo con antebrazo, palma y 5 dedos anatómicos articulados."""
    wx, wy = wrist_pt
    px, py = palm_pt

    # 1. Antebrazo extendiéndose desde la muñeca hacia afuera
    dx, dy = wx - px, wy - py
    dist = max(1.0, math.hypot(dx, dy))
    nx, ny = dx / dist, dy / dist
    arm_end = (int(wx + nx * 600), int(wy + ny * 600))
    cv2.line(frame, wrist_pt, arm_end, clothing_color, thickness=95)

    # 2. Conexión muñeca-palma
    cv2.line(frame, wrist_pt, palm_pt, skin_color, thickness=80)
    cv2.circle(frame, palm_pt, 45, skin_color, -1)
    cv2.circle(frame, wrist_pt, 42, skin_color, -1)

    # 3. Vector perpendicular para orientar los dedos
    perp_x, perp_y = -ny, nx

    # 4. Dedos (pulgar, índice, medio, anular, meñique)
    finger_lengths = [65, 95, 105, 95, 75]
    finger_offsets = [-0.65, -0.30, 0.0, 0.30, 0.60]

    for idx, (flen, foff) in enumerate(zip(finger_lengths, finger_offsets)):
        if pinch and idx in (0, 1):
            # Pulgar e índice tocándose en la punta
            target_x = int(px - nx * 85 + perp_x * -10)
            target_y = int(py - ny * 85 + perp_y * -10)
            cv2.line(frame, palm_pt, (target_x, target_y), skin_color, thickness=16)
            cv2.circle(frame, (target_x, target_y), 9, skin_color, -1)
            continue

        base_x = int(px - nx * 20 + perp_x * foff * 50 * finger_spread)
        base_y = int(py - ny * 20 + perp_y * foff * 50 * finger_spread)

        tip_x = int(base_x - nx * flen + perp_x * foff * 35 * finger_spread)
        tip_y = int(base_y - ny * flen + perp_y * foff * 35 * finger_spread)

        cv2.line(frame, palm_pt, (base_x, base_y), skin_color, thickness=20)
        cv2.line(frame, (base_x, base_y), (tip_x, tip_y), skin_color, thickness=16)
        cv2.circle(frame, (tip_x, tip_y), 8, skin_color, -1)


def generate_clip_a_hand_entry() -> Path:
    """(a) Mano entrando por cada uno de los 4 bordes del encuadre."""
    path = OUTPUT_DIR / "clip_a_hand_entry_borders.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = 120  # 4 segundos (30 cuadros por borde)
    for f in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)  # Fondo gris oscuro/pared
        edge_phase = f // 30
        t = (f % 30) / 30.0

        if edge_phase == 0:  # Entrada por borde DERECHO
            px = int(WIDTH + 150 - t * 450)
            py = 360
            wx = px + 140
            wy = py
        elif edge_phase == 1:  # Entrada por borde IZQUIERDO
            px = int(-150 + t * 450)
            py = 360
            wx = px - 140
            wy = py
        elif edge_phase == 2:  # Entrada por borde INFERIOR
            px = 640
            py = int(HEIGHT + 150 - t * 450)
            wx = px
            wy = py + 140
        else:  # Entrada por borde SUPERIOR
            px = 640
            py = int(-150 + t * 450)
            wx = px
            wy = py - 140

        draw_realistic_hand(frame, (wx, wy), (px, py))
        out.write(frame)

    out.release()
    return path


def generate_clip_b_hand_slow_fast() -> Path:
    """(b) Mano moviéndose a velocidad lenta y luego rápida con pellizcos."""
    path = OUTPUT_DIR / "clip_b_hand_slow_fast.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = 120
    for f in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 32, dtype=np.uint8)

        if f < 60:
            # Fase lenta (~100 px/seg)
            t = f / 60.0
            px = int(300 + t * 250)
            py = int(350 + math.sin(t * math.pi * 2) * 50)
            pinch = 25 <= f <= 45
        else:
            # Fase rápida (~900 px/seg)
            t = (f - 60) / 60.0
            px = int(550 + math.sin(t * math.pi * 4) * 450)
            py = int(350 + math.cos(t * math.pi * 4) * 150)
            pinch = (f % 15) < 7

        wx = px + 150
        wy = py + 30
        draw_realistic_hand(frame, (wx, wy), (px, py), pinch=pinch)
        out.write(frame)

    out.release()
    return path


def generate_clip_c_robot_poster_static() -> Path:
    """(c) Escena con silueta humanoide estática/robot SIN humano real (falsos positivos)."""
    path = OUTPUT_DIR / "clip_c_robot_poster_static.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))

    for _ in range(90):  # 3 segundos estáticos
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)

        # Dibujar poster/maniquí humanoide metálico estático en el centro
        cv2.rectangle(frame, (480, 200), (800, 680), (70, 75, 80), -1)  # Torso
        cv2.circle(frame, (640, 140), 60, (90, 95, 100), -1)  # Cabeza
        cv2.line(frame, (480, 260), (360, 480), (70, 75, 80), 35)  # Brazo izquierdo
        cv2.line(frame, (800, 260), (920, 480), (70, 75, 80), 35)  # Brazo derecho
        cv2.putText(frame, "POSTER FIGURA HUMANOIDE", (420, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out.write(frame)

    out.release()
    return path


def generate_clip_d_empty_wall_lighting() -> Path:
    """(d) Pared vacía con cambios de iluminación global y gradiente."""
    path = OUTPUT_DIR / "clip_d_empty_wall_lighting.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))

    for f in range(90):  # 3 segundos
        t = f / 90.0
        # Simular variación de luz y autoexposición acotada a rango uint8 válido
        base_lum = int(np.clip(50 + 30 * math.sin(t * math.pi * 3), 15, 200))
        frame = np.full((HEIGHT, WIDTH, 3), base_lum, dtype=np.uint8)

        # Gradiente simulando luz del proyector en la pared
        spot_val = int(np.clip(base_lum + 30 * math.sin(t * math.pi * 2), 10, 240))
        cv2.circle(frame, (640, 360), 300, (spot_val, spot_val, spot_val), -1)
        frame = cv2.GaussianBlur(frame, (99, 99), 30)

        out.write(frame)

    out.release()
    return path


def generate_clip_e_two_hands() -> Path:
    """(e) Dos manos entrando simultáneamente por bordes opuestos."""
    path = OUTPUT_DIR / "clip_e_two_hands.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))

    for f in range(90):  # 3 segundos
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        t = min(1.0, f / 45.0)

        # Mano derecha del operador entrando por la derecha
        p1_x = int(WIDTH + 100 - t * 380)
        p1_y = 300
        draw_realistic_hand(frame, (p1_x + 140, p1_y), (p1_x, p1_y))

        # Mano izquierda del operador entrando por la izquierda
        p2_x = int(-100 + t * 380)
        p2_y = 420
        draw_realistic_hand(frame, (p2_x - 140, p2_y), (p2_x, p2_y))

        out.write(frame)

    out.release()
    return path


def generate_clip_f_torso_present() -> Path:
    """(f) Torso y cabeza visibles mientras un brazo entra e interactúa."""
    path = OUTPUT_DIR / "clip_f_torso_present.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))

    for f in range(90):  # 3 segundos
        frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        t = f / 90.0

        # Torso y cabeza en la mitad izquierda
        body_color = (80, 50, 40)
        cv2.rectangle(frame, (250, 240), (550, 720), body_color, -1)  # Torso
        cv2.circle(frame, (400, 150), 65, (165, 195, 225), -1)  # Cabeza

        # Brazo saliendo del hombro hacia la derecha de la escena
        hx, hy = 480, 320  # Hombro
        arm_t = math.sin(t * math.pi * 2) * 80
        px = int(820 + arm_t)
        py = int(360 + math.cos(t * math.pi * 2) * 50)
        wx = int(px - 140)
        wy = py

        cv2.line(frame, (hx, hy), (wx, wy), body_color, thickness=90)
        draw_realistic_hand(frame, (wx, wy), (px, py), pinch=(f % 20 < 10))

        out.write(frame)

    out.release()
    return path


def main() -> None:
    print(f"🎬 Generando 6 clips de prueba sintéticos en: {OUTPUT_DIR}")
    clips = [
        ("a", generate_clip_a_hand_entry()),
        ("b", generate_clip_b_hand_slow_fast()),
        ("c", generate_clip_c_robot_poster_static()),
        ("d", generate_clip_d_empty_wall_lighting()),
        ("e", generate_clip_e_two_hands()),
        ("f", generate_clip_f_torso_present()),
    ]
    for tag, p in clips:
        size_kb = p.stat().st_size / 1024
        print(f"  ✓ ({tag}) {p.name} [{size_kb:.1f} KB]")
    print("✅ Generación completada con éxito.")


if __name__ == "__main__":
    main()
