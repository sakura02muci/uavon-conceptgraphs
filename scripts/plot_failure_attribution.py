"""Summarize evaluation failures using graph, navigation, and collision evidence.

The graphic is deliberately episode-level: its labels are rule-based, traceable
diagnoses rather than a claim that the benchmark supplies ground-truth causes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def classify(episode: dict) -> tuple[str, str]:
    """Return a primary diagnosis and a short evidence string."""
    if episode.get("success"):
        return "Success", "target contact"

    collisions = int(episode.get("collision_count", 0))
    min_distance = float(episode.get("min_distance_to_goal", float("inf")))
    graph = episode.get("graph_goal_diagnostics") or {}
    graph_distance = graph.get("nearest_target_like_distance")
    graph_distance = float(graph_distance) if graph_distance is not None else float("inf")

    if collisions >= 8:
        return "Collision recovery loop", f"{collisions} collisions"
    if min_distance <= 10.0:
        return "Close, but executor did not finish", f"minimum {min_distance:.1f} m"
    if graph_distance > 10.0:
        return "Target node mislocalized", f"best graph target {graph_distance:.1f} m from goal"
    return "Navigation / node-ranking failure", (
        f"graph target {graph_distance:.1f} m, UAV minimum {min_distance:.1f} m"
    )


COLORS = {
    "Success": "#16a34a",
    "Collision recovery loop": "#dc2626",
    "Close, but executor did not finish": "#f59e0b",
    "Target node mislocalized": "#7c3aed",
    "Navigation / node-ranking failure": "#2563eb",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rows = []
    for episode in episodes:
        graph = episode.get("graph_goal_diagnostics") or {}
        reason, evidence = classify(episode)
        rows.append({
            "episode": str(episode["episode_id"]),
            "target": str(episode["target"]),
            "reason": reason,
            "evidence": evidence,
            "minimum": float(episode["min_distance_to_goal"]),
            "graph_distance": float(graph.get("nearest_target_like_distance") or np.nan),
            "collisions": int(episode.get("collision_count", 0)),
        })

    labels = [f"{r['episode']}\n{r['target']}" for r in rows]
    x = np.arange(len(rows))
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.25, 1), hspace=0.35)
    ax = fig.add_subplot(grid[0])
    min_dist = np.array([r["minimum"] for r in rows])
    graph_dist = np.array([r["graph_distance"] for r in rows])
    colors = [COLORS[r["reason"]] for r in rows]
    ax.bar(x, min_dist, color=colors, width=0.65, alpha=0.9, label="UAV minimum distance to true goal")
    valid = ~np.isnan(graph_dist)
    ax.scatter(x[valid], graph_dist[valid], marker="D", color="#111827", s=52,
               zorder=3, label="closest target-like graph node to true goal")
    for i, row in enumerate(rows):
        ax.text(i, min_dist[i] + 0.8, f"{row['collisions']} collision" + ("s" if row["collisions"] != 1 else ""),
                ha="center", va="bottom", fontsize=9)
    ax.axhline(10, color="#64748b", linestyle="--", linewidth=1, label="10 m diagnostic threshold")
    ax.set_ylim(0, max(np.nanmax(np.r_[min_dist, graph_dist]) + 8, 30))
    ax.set_xticks(x, labels)
    ax.set_ylabel("distance to true goal (m)")
    ax.set_title("Visible-target evaluation: where each episode failed", loc="left", fontsize=16, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, fontsize=9, loc="upper left")

    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    columns = ["Episode / target", "Primary diagnosis", "Trace evidence"]
    cells = [[f"{r['episode']} · {r['target']}", r["reason"], r["evidence"]] for r in rows]
    table = table_ax.table(cellText=cells, colLabels=columns, cellLoc="left", colLoc="left",
                           loc="center", colWidths=[0.18, 0.31, 0.51])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        elif col == 1:
            cell.set_facecolor(COLORS[rows[row - 1]["reason"]] + "22")

    fig.text(0.125, 0.015,
             "Diagnosis rules: collision loop >=8 collisions; close-but-unfinished <=10 m; "
             "otherwise a graph target >10 m from the true goal is treated as mislocalization.",
             fontsize=9, color="#475569")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
