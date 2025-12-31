from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# Ensure matplotlib uses a writable config dir
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt


def render_round_metrics(
    *,
    output_path: Path,
    turns: List[int],
    agents: List[str],
    series: Dict[str, Dict[str, List[int]]],
) -> None:
    """Render a multi-panel metrics chart for a full round."""
    metric_titles = [
        ("total_welfare", "Total Welfare"),
        ("welfare_this_turn", "Welfare This Turn"),
        ("attacks", "Attacks (Mils Committed)"),
        ("attacks_received", "Attacks Received (Mils)"),
        ("army_size", "Army Size (End of Turn)"),
        ("territories", "Territories Owned"),
        ("mils_destroyed", "Mils Destroyed (Attacker Losses)"),
        ("mils_disbanded", "Mils Disbanded (Upkeep)"),
        ("trade_sent", "Trade Sent (Grants Paid)"),
        ("trade_welfare_received", "Trade Welfare Received"),
    ]

    fig, axes = plt.subplots(len(metric_titles), 1, figsize=(10, 18), sharex=True)

    for idx, (key, title) in enumerate(metric_titles):
        ax = axes[idx]
        for agent in agents:
            ax.plot(turns, series[key][agent], marker="o", label=agent)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left", ncol=2, fontsize=8)

    axes[-1].set_xlabel("Turn")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
