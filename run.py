#!/usr/bin/env python3
"""
Touchless Projection Mapping — Orquestador de Ejecución Nativo Profesional
==========================================================================
Gestiona el ciclo de vida coordinado de la suite completa en el Host:
  [1] Backend Hardware/IoT y Editor de Polígonos (FastAPI en puerto 8000)
  [2] Motor de Visión Neuronal y Pipeline CUDA   (Puerto 5051)
  [3] Apertura automática y sincronizada de interfaces en el navegador
  [4] Manejo robusto de señales (Ctrl+C / SIGINT) con apagado limpio sin procesos huérfanos.
"""
from __future__ import annotations

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import List, Optional

# Reconfigurar salida para UTF-8 limpio en terminales Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

# Paleta ANSI de alta fidelidad
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner() -> None:
    """Imprime el encabezado visual de grado estudio Pro-AV."""
    print(f"{CYAN}{BOLD}")
    print("=" * 74)
    print("  🖐️  TOUCHLESS PROJECTION MAPPING — SUITE PROFESIONAL DE ESTUDIO")
    print("=" * 74)
    print(f"{RESET}{DIM}  Arquitectura: src-layout | Motor: CUDA + PyTorch | Control: WebSockets + IoT{RESET}\n")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Comprueba si un puerto TCP ya se encuentra ocupado."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def check_virtualenv() -> None:
    """Verifica si se ejecuta bajo el entorno virtual .venv; si no, re-ejecuta con .venv."""
    in_venv = sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ
    venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    if not in_venv and venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        print(f"  {YELLOW}[*] Reiniciando orquestador bajo el entorno virtual (.venv)...{RESET}")
        cmd = [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


def run_preflight_checks(backend_port: int, vision_port: int) -> bool:
    """Ejecuta diagnósticos previos para prevenir errores de hardware o puertos."""
    print(f"{BOLD}[1/4] Diagnósticos del Sistema y Hardware:{RESET}")

    # 1. GPU y Aceleración CUDA
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  {GREEN}✓ GPU NVIDIA Detectada:{RESET} {dev_name} ({vram_gb:.1f} GB VRAM, CUDA {torch.version.cuda})")
        else:
            print(f"  {YELLOW}⚠ GPU CUDA no disponible:{RESET} Ejecutando en CPU (Modo de rendimiento reducido)")
    except ImportError:
        print(f"  {YELLOW}⚠ PyTorch no disponible:{RESET} Ejecutando comprobación básica")

    # 2. Comprobación de puertos de red
    ports_ok = True
    if is_port_in_use(backend_port):
        print(f"  {RED}✗ Error:{RESET} El puerto {backend_port} (Backend) ya está ocupado por otra aplicación.")
        ports_ok = False
    else:
        print(f"  {GREEN}✓ Puerto {backend_port} (Backend IoT):{RESET} Libre y listo")

    if is_port_in_use(vision_port):
        print(f"  {RED}✗ Error:{RESET} El puerto {vision_port} (Visión) ya está ocupado por otra aplicación.")
        ports_ok = False
    else:
        print(f"  {GREEN}✓ Puerto {vision_port} (Visión Neuronal):{RESET} Libre y listo")

    print()
    return ports_ok


class ProcessManager:
    """Administrador de subprocesos concurrentes con parada limpia garantizada."""

    def __init__(self) -> None:
        self.processes: List[subprocess.Popen] = []
        self._is_shutting_down = False

    def spawn(self, cmd: List[str], name: str, env: Optional[dict] = None) -> subprocess.Popen:
        """Inicia un subproceso hijo registrándolo para limpieza automática."""
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        proc_env["PYTHONUNBUFFERED"] = "1"
        proc_env["PYTHONIOENCODING"] = "utf-8"
        proc_env["PYTHONPATH"] = str(SRC_DIR)

        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            env=proc_env,
        )
        self.processes.append(p)
        return p

    def shutdown(self) -> None:
        """Detiene de forma ordenada todos los subprocesos en ejecución."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        print(f"\n{YELLOW}[*] Deteniendo servicios del sistema de proyección...{RESET}")

        for p in self.processes:
            if p.poll() is None:
                p.terminate()

        # Esperar hasta 2.5s a que cierren amistosamente
        deadline = time.time() + 2.5
        for p in self.processes:
            while time.time() < deadline and p.poll() is None:
                time.sleep(0.05)

        # Forzar terminación en Windows si algún proceso quedó suspendido
        for p in self.processes:
            if p.poll() is None:
                try:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        p.kill()
                except Exception:
                    pass

        print(f"{GREEN}✓ Todos los servicios se han detenido correctamente. Puertos liberados.{RESET}")


def wait_for_service(url: str, timeout_s: float = 12.0) -> bool:
    """Espera activamente a que un endpoint HTTP responda con código 200 OK."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OrchestratorHealthCheck"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lanzador Orquestador Profesional Touchless Projection Mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("camera_index", nargs="?", type=int, default=None, help="Índice de cámara (0, 1, 2...)")
    parser.add_argument("--source", type=str, default="", help="Ruta a clip .mp4 offline para pruebas reproducibles")
    parser.add_argument("--model", type=str, default="rvm_mobilenet", help="Modelo neuronal inicial")
    parser.add_argument("--backend-port", type=int, default=8000, help="Puerto HTTP del Backend IoT y Editor")
    parser.add_argument("--vision-port", type=int, default=5051, help="Puerto HTTP de la Visión y Proyector")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host de escucha")
    parser.add_argument("--no-browser", action="store_true", help="No abrir automáticamente las pestañas del navegador")
    parser.add_argument("--native-projector", action="store_true", help="Abrir ventana nativa OpenCV en monitor secundario")
    parser.add_argument("--show-window", action="store_true", help="Mostrar ventana de depuración OpenCV")
    parser.add_argument("--check-only", action="store_true", help="Ejecutar únicamente comprobaciones de hardware y salir")
    args = parser.parse_args()

    check_virtualenv()
    print_banner()

    if not run_preflight_checks(args.backend_port, args.vision_port):
        sys.exit(1)

    if args.check_only:
        print(f"{GREEN}✓ Diagnósticos finalizados con éxito.{RESET}")
        sys.exit(0)

    mgr = ProcessManager()

    def signal_handler(signum, frame):
        mgr.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(mgr.shutdown)

    # 1. Iniciar Servidor Backend (FastAPI / IoT en puerto 8000)
    print(f"{BOLD}[2/4] Levantando Backend de Dispositivos y Editor (Puerto {args.backend_port})...{RESET}")
    backend_cmd = [
        sys.executable,
        "-m", "backend",
        "--devices", "config/devices.dev.yaml",
        "--objects", "config/objects.json",
        "--port", str(args.backend_port),
        "--host", args.host,
    ]
    mgr.spawn(backend_cmd, "Backend")

    # 2. Iniciar Motor de Visión Neuronal (Puerto 5051)
    print(f"{BOLD}[3/4] Levantando Motor de Visión Acelerada por CUDA (Puerto {args.vision_port})...{RESET}")
    vision_cmd = [
        sys.executable,
        "-m", "vision",
        "--port", str(args.vision_port),
        "--host", args.host,
        "--model", args.model,
    ]
    if args.source:
        vision_cmd.extend(["--source", args.source])
    elif args.camera_index is not None:
        vision_cmd.append(str(args.camera_index))
    if args.native_projector:
        vision_cmd.append("--native-projector")
    if args.show_window:
        vision_cmd.append("--show-window")

    p_vision = mgr.spawn(vision_cmd, "Vision")

    # 3. Esperar confirmación de salud y abrir interfaces web
    print(f"{BOLD}[4/4] Sincronizando interfaces y endpoints...{RESET}")
    backend_ready = wait_for_service(f"http://127.0.0.1:{args.backend_port}/api/objects", timeout_s=10.0)
    vision_ready = wait_for_service(f"http://127.0.0.1:{args.vision_port}/api/status", timeout_s=15.0)

    if backend_ready and vision_ready:
        print(f"\n{GREEN}{BOLD}✓ SISTEMA TOUCHLESS EN LÍNEA Y OPERATIVO:{RESET}")
        print(f"  {CYAN}🖥️  Panel del Operador:{RESET}    http://localhost:{args.vision_port}/controller.html")
        print(f"  {CYAN}📽️  Pared de Proyección:{RESET}   http://localhost:{args.vision_port}/projector.html")
        print(f"  {CYAN}📐 Editor de Polígonos:{RESET}   http://localhost:{args.backend_port}/editor\n")

        if not args.no_browser:
            time.sleep(0.5)
            webbrowser.open(f"http://localhost:{args.vision_port}/controller.html")
            webbrowser.open(f"http://localhost:{args.vision_port}/projector.html")
    else:
        print(f"  {YELLOW}⚠ Advertencia:{RESET} Uno de los servicios tardó más de lo esperado en iniciar.")

    print(f"{DIM}Presiona [Ctrl + C] en cualquier momento para detener todos los servicios de forma limpia.{RESET}\n")

    # Mantener el hilo principal vivo mientras los subprocesos continúen
    try:
        while True:
            for p in mgr.processes:
                ret = p.poll()
                if ret is not None and ret != 0:
                    print(f"\n{RED}[!] Un subproceso terminó inesperadamente con código {ret}.{RESET}")
                    mgr.shutdown()
                    sys.exit(ret)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.shutdown()


if __name__ == "__main__":
    main()
