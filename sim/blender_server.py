from __future__ import annotations

"""
Minimal Blender render service.

Run inside Blender:

    blender -b -P sim/blender_server.py -- --host 127.0.0.1 --port 8765

POST /render JSON body example:
{
  "csv": "D:/robot_view/ore/outputs/dataset_rtn/val/no_maneuver/sample_00400.csv",
  "out": "D:/robot_view/ore/outputs/validation_runs/demo/original_render"
}

Optional fields: obj, resx, resy, fps, fov_deg, scale, emission
"""

import argparse
import json
import runpy
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

RENDER_SCRIPT = Path(config.BLENDER_SCRIPT)


def _build_render_argv(payload: dict[str, object]) -> list[str]:
    csv_path = str(payload.get("csv", "")).strip()
    out_dir = str(payload.get("out", "")).strip()
    if not csv_path:
        raise ValueError("missing required field: csv")
    if not out_dir:
        raise ValueError("missing required field: out")

    obj_path = str(payload.get("obj", config.TARGET_MODEL))
    resx = int(payload.get("resx", config.IMG_W))
    resy = int(payload.get("resy", config.IMG_H))
    fps = int(payload.get("fps", config.RENDER_FPS))
    fov_deg = float(payload.get("fov_deg", config.RENDER_FOV_DEG))
    scale = float(payload.get("scale", config.RENDER_SCALE))
    emission = float(payload.get("emission", config.RENDER_EMISSION))

    return [
        str(RENDER_SCRIPT),
        "--",
        "--csv",
        csv_path,
        "--obj",
        obj_path,
        "--out",
        out_dir,
        "--resx",
        str(resx),
        "--resy",
        str(resy),
        "--fps",
        str(fps),
        "--fov_deg",
        str(fov_deg),
        "--scale",
        str(scale),
        "--emission",
        str(emission),
    ]


def _render_once(payload: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    old_argv = list(sys.argv)
    try:
        sys.argv = _build_render_argv(payload)
        runpy.run_path(str(RENDER_SCRIPT), run_name="__main__")
    finally:
        sys.argv = old_argv
    elapsed = time.perf_counter() - start
    return {"elapsed_sec": float(elapsed)}


class BlenderRenderHandler(BaseHTTPRequestHandler):
    server_version = "BlenderRenderServer/1.0"

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"status": "ok", "service": "blender-render-server"})
            return
        self._send_json(404, {"status": "error", "error": "not_found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/render":
            self._send_json(404, {"status": "error", "error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json(400, {"status": "error", "error": "empty_request_body"})
            return

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(400, {"status": "error", "error": "invalid_json"})
            return

        try:
            result = _render_once(payload if isinstance(payload, dict) else {})
            self._send_json(200, {"status": "ok", **result})
        except Exception as exc:
            self._send_json(500, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: object) -> None:
        print("[BlenderServer] " + (fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Blender render HTTP service")
    parser.add_argument("--host", default=str(getattr(config, "BLENDER_SERVER_HOST", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(getattr(config, "BLENDER_SERVER_PORT", 8765)))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    server = HTTPServer((args.host, args.port), BlenderRenderHandler)
    print(f"Blender render server listening on http://{args.host}:{args.port}/render")
    print(f"Render backend script: {RENDER_SCRIPT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
