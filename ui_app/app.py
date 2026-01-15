from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

def _build_capital_pies(payload: Dict[str, object], turn_idx: int) -> Dict[str, Dict[str, float]]:
    ledger_vars = payload.get("ledger_vars") or []
    data = payload.get("data") or []
    agents = payload.get("agents") or []
    capitals = payload.get("capitals") or {}
    constants = payload.get("constants") or {}
    if not isinstance(ledger_vars, list) or not isinstance(data, list) or not agents or not capitals:
        return {}
    required = [
        "gross_income",
        "effective_income",
        "damage_received",
        "upkeep_cost",
        "mils_purchased",
        "money_grants_received",
    ]
    if any(key not in ledger_vars for key in required):
        return {}
    idx = {key: ledger_vars.index(key) for key in required}
    trade_factor = float(constants.get("c_trade_factor", 1.0))
    purchase_price = float(constants.get("c_mil_purchase_price", 0.0))
    pies: Dict[str, Dict[str, float]] = {}
    for agent_idx, agent in enumerate(agents):
        if agent_idx >= len(data):
            continue
        rows = data[agent_idx]
        if not isinstance(rows, list) or turn_idx >= len(rows):
            continue
        row = rows[turn_idx]
        if not isinstance(row, list):
            continue
        if len(row) <= max(idx.values()):
            continue
        gross = float(row[idx["gross_income"]])
        effective = float(row[idx["effective_income"]])
        damage = float(row[idx["damage_received"]])
        upkeep = float(row[idx["upkeep_cost"]])
        mils_purchased = float(row[idx["mils_purchased"]])
        grants_received = float(row[idx["money_grants_received"]])
        total = gross + grants_received * trade_factor
        if total <= 0:
            continue
        lost = max(gross - effective, 0.0)
        purchases = max(mils_purchased * purchase_price, 0.0)
        upkeep = max(upkeep, 0.0)
        damage = max(damage, 0.0)
        welfare = max(total - (lost + purchases + upkeep + damage), 0.0)
        capital = capitals.get(agent)
        if not capital:
            continue
        pies[capital] = {
            "lost": lost,
            "purchases": purchases,
            "upkeep": upkeep,
            "damage": damage,
            "welfare": welfare,
        }
    return pies


class RoundDataHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _round_data_dir(self) -> Path:
        return BASE_DIR.parent / "round_data"

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

        if parsed.path == "/api/experiments":
            exp_dir = BASE_DIR.parent / "experiments"
            if not exp_dir.exists():
                self._send_json({"files": []})
                return
            files = sorted(p.name for p in exp_dir.iterdir() if p.is_file() and p.suffix == ".json")
            self._send_json({"files": files})
            return

        if parsed.path == "/api/experiment":
            params = parse_qs(parsed.query)
            file_name = params.get("file", [""])[0]
            safe = os.path.basename(unquote(file_name))
            exp_path = BASE_DIR.parent / "experiments" / safe
            if not exp_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Experiment file not found")
                return
            try:
                data = json.loads(exp_path.read_text(encoding="utf-8"))
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Invalid experiment file")
                return

            from analyze_experiment import summarize_experiment

            summary = summarize_experiment(data)
            self._send_json({"summary": summary, "raw": data})
            return

        if parsed.path == "/api/experiment_chart":
            params = parse_qs(parsed.query)
            file_name = params.get("file", [""])[0]
            chart_type = params.get("type", ["avg"])[0]
            safe = os.path.basename(unquote(file_name))
            exp_path = BASE_DIR.parent / "experiments" / safe
            if not exp_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Experiment file not found")
                return
            try:
                data = json.loads(exp_path.read_text(encoding="utf-8"))
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Invalid experiment file")
                return

            from analyze_experiment import (
                compute_modifier_stats,
                render_bar_png,
                render_box_png,
            )

            stats = compute_modifier_stats(data)
            if chart_type == "wins":
                values = {k: float(v.wins) for k, v in stats.items()}
                img = render_bar_png(values, title="Wins by Modifier")
            elif chart_type == "dist":
                img = render_box_png(stats, title="Welfare Distribution by Modifier")
            else:
                values = {k: v.average() for k, v in stats.items()}
                img = render_bar_png(values, title="Average Welfare by Modifier")

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(img)))
            self.end_headers()
            self.wfile.write(img)
            return

        if parsed.path == "/api/map":
            params = parse_qs(parsed.query)
            file_name = params.get("file", [""])[0]
            turn_raw = params.get("turn", [""])[0]
            data_path = self._get_data_file(unquote(file_name))
            if data_path is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Data file not found")
                return
            try:
                payload = json.loads(data_path.read_text(encoding="utf-8"))
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Invalid data file")
                return

            turns = payload.get("turns", [])
            if not turns:
                self.send_error(HTTPStatus.BAD_REQUEST, "No turns available")
                return
            try:
                turn_val = int(turn_raw)
            except Exception:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid turn")
                return
            if turn_val in turns:
                turn_idx = turns.index(turn_val)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Turn not found")
                return

            territory_names = payload.get("territory_names", [])
            territory_positions = payload.get("territory_positions", {})
            territory_owners = payload.get("territory_owners", [])
            territory_resources = payload.get("territory_resources", {})
            agents = payload.get("agents", [])
            if not territory_names or not territory_owners:
                self.send_error(HTTPStatus.BAD_REQUEST, "No territory data")
                return

            positions = {
                name: tuple(territory_positions.get(name, (0, 0)))
                for name in territory_names
            }
            owners = territory_owners[turn_idx]
            palette = [
                "#34a39a",
                "#e98467",
                "#5f7a75",
                "#f4b073",
                "#93bd8a",
                "#c995a7",
                "#6f879e",
                "#f7c372",
                "#5aa39a",
                "#f58a4b",
            ]
            owner_colors = {
                agent: palette[idx % len(palette)] for idx, agent in enumerate(agents)
            }

            from peacegame.territory_graph import PIE_COLORS, PIE_ORDER
            from peacegame.worldmap_renderer import render_world_map_png

            capital_pies = _build_capital_pies(payload, turn_idx)

            owner_by_name = {name: owner for name, owner in zip(territory_names, owners)}
            img = render_world_map_png(
                owner_by_name=owner_by_name,
                owner_colors=owner_colors,
                territory_resources=territory_resources,
                capital_pies=capital_pies,
                pie_colors=PIE_COLORS,
                pie_order=PIE_ORDER,
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(img)))
            self.end_headers()
            self.wfile.write(img)
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
