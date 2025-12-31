from __future__ import annotations

import argparse
import json
import os
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class RoundDataHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _round_data_dir(self) -> Path:
        return Path(os.getcwd()) / "round_data"

    def _get_data_file(self, name: str) -> Optional[Path]:
        safe = os.path.basename(name)
        if safe != name or not safe.endswith(".json"):
            return None
        path = self._round_data_dir() / safe
        if not path.is_file():
            return None
        return path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/files":
            data_dir = self._round_data_dir()
            if not data_dir.exists():
                self._send_json({"files": []})
                return
            files = sorted(p.name for p in data_dir.iterdir() if p.is_file() and p.name.endswith(".json"))
            self._send_json({"files": files})
            return

        if parsed.path.startswith("/api/data/"):
            raw_name = unquote(parsed.path[len("/api/data/") :])
            data_path = self._get_data_file(raw_name)
            if data_path is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Data file not found")
                return
            try:
                payload = json.loads(data_path.read_text(encoding="utf-8"))
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Invalid data file")
                return
            self._send_json(payload)
            return

        if parsed.path in ("", "/"):
            self.path = "/index.html"
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Round data viewer")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    handler = partial(RoundDataHandler, directory=str(STATIC_DIR))
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Round data viewer running at http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
