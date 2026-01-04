from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt


@dataclass
class ModifierStats:
    total_welfare: int = 0
    rounds: int = 0
    wins: int = 0
    welfare_samples: List[int] | None = None

    def __post_init__(self) -> None:
        if self.welfare_samples is None:
            self.welfare_samples = []

    def add_sample(self, welfare: int, won: bool) -> None:
        self.total_welfare += welfare
        self.rounds += 1
        if won:
            self.wins += 1
        self.welfare_samples.append(welfare)

    def average(self) -> float:
        return self.total_welfare / max(self.rounds, 1)

    def min(self) -> int:
        return min(self.welfare_samples) if self.welfare_samples else 0

    def max(self) -> int:
        return max(self.welfare_samples) if self.welfare_samples else 0

    def median(self) -> float:
        return float(median(self.welfare_samples)) if self.welfare_samples else 0.0


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_modifier_stats(data: Dict[str, Any]) -> Dict[str, ModifierStats]:
    stats: Dict[str, ModifierStats] = {}
    results = data.get("results", [])
    for result in results:
        modifiers = result.get("modifiers", {})
        welfare = result.get("welfare", {})
        ranking = result.get("ranking", [])
        winner = ranking[0][0] if ranking else None
        for agent, mod in modifiers.items():
            mod_name = mod or "none"
            ms = stats.setdefault(mod_name, ModifierStats())
            score = int(welfare.get(agent, 0))
            ms.add_sample(score, won=(agent == winner))
    return stats


def summarize_experiment(data: Dict[str, Any]) -> str:
    name = data.get("name", "experiment")
    setup = data.get("setup", "")
    rounds = int(data.get("rounds", 0))
    num_turns = int(data.get("num_turns", 0))
    results = data.get("results", [])
    stats = compute_modifier_stats(data)

    lines: List[str] = []
    lines.append(f"Experiment: {name}")
    lines.append(f"Setup: {setup}")
    lines.append(f"Rounds: {rounds}")
    lines.append(f"Turns per round: {num_turns}")
    lines.append("")
    lines.append("Round winners:")
    for result in results:
        round_idx = result.get("round")
        ranking = result.get("ranking", [])
        if not ranking:
            continue
        winner, score, _rank = ranking[0]
        mod = result.get("modifiers", {}).get(winner, "none")
        lines.append(f"- Round {round_idx}: {winner} ({mod}) welfare={score}")
    lines.append("")
    lines.append("Modifier summary (sorted by average welfare):")
    for mod, ms in sorted(stats.items(), key=lambda x: (-x[1].average(), x[0])):
        lines.append(
            f"- {mod}: rounds={ms.rounds}, wins={ms.wins}, "
            f"avg={ms.average():.2f}, total={ms.total_welfare}, "
            f"min={ms.min()}, median={ms.median():.1f}, max={ms.max()}"
        )
    return "\n".join(lines)


def render_bar_png(values: Dict[str, float], *, title: str) -> bytes:
    labels = list(values.keys())
    scores = [values[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, scores, color="#5aa39a")
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def render_box_png(stats: Dict[str, ModifierStats], *, title: str) -> bytes:
    labels = list(stats.keys())
    data = [stats[k].welfare_samples for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showfliers=False, whis=(0, 100))
    ax.set_title(title)
    ax.set_ylabel("Welfare")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Welfare", fontsize=11)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("path", help="Path to experiment JSON")
    parser.add_argument(
        "--write-files",
        action="store_true",
        help="Write summary/plots to disk instead of stdout only",
    )
    args = parser.parse_args()

    data = _load_json(args.path)
    stats = compute_modifier_stats(data)
    summary = summarize_experiment(data)

    if args.write_files:
        stem = Path(args.path).stem
        summary_path = Path("experiments") / f"{stem}_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")

        avg_values = {k: v.average() for k, v in stats.items()}
        win_values = {k: float(v.wins) for k, v in stats.items()}

        out_dir = Path("visualizations")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}_avg.png").write_bytes(
            render_bar_png(avg_values, title="Average Welfare by Modifier")
        )
        (out_dir / f"{stem}_wins.png").write_bytes(
            render_bar_png(win_values, title="Wins by Modifier")
        )
        (out_dir / f"{stem}_dist.png").write_bytes(
            render_box_png(stats, title="Welfare Distribution by Modifier")
        )
        print(f"Summary written to {summary_path}")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
