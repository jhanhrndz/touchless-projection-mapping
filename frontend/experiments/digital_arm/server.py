import http.server
import socketserver
import json
import os
from pathlib import Path

PORT = 5051
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(DIRECTORY).parent.parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        # Serve the objects file directly from the data folder to avoid CORS issues
        if self.path == '/api/objects':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            objects_path = ROOT_DIR / "data" / "objects.json"
            if objects_path.exists():
                with open(objects_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(b"[]")
        else:
            super().do_GET()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("="*60)
        print(f"EXPERIMENTO: BRAZO DIGITAL / PORTAL (Iniciado en el puerto {PORT})")
        print("="*60)
        print(f"Abre tu navegador en: http://localhost:{PORT}/")
        print("Asegurate de tener tu backend principal (docker compose up) corriendo en el puerto 8080.")
        print("Presiona Ctrl+C para detener el servidor.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")
