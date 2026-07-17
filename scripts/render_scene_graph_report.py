"""Render current UAV-ON scene graph JSON files for Markdown reports."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render_graphs(graph_dir: Path, output: Path) -> None:
    graph_files = sorted(graph_dir.glob("episode_*_graph.json"))
    if not graph_files:
        raise FileNotFoundError(f"No episode graph JSON files found in {graph_dir}")

    fig, axes = plt.subplots(1, len(graph_files), figsize=(7 * len(graph_files), 6), squeeze=False)
    for ax, graph_file in zip(axes[0], graph_files):
        graph = json.loads(graph_file.read_text(encoding="utf-8"))
        nodes = graph.get("nodes", [])
        node_by_id = {str(node["node_id"]): node for node in nodes}
        labels = Counter(str(node.get("label", "unknown")) for node in nodes)
        palette = plt.get_cmap("tab20")
        label_order = {label: index for index, (label, _) in enumerate(labels.most_common())}

        for edge in graph.get("edges", []):
            source = node_by_id.get(str(edge.get("source")))
            target = node_by_id.get(str(edge.get("target")))
            if source is None or target is None:
                continue
            start = np.asarray(source["centroid"], dtype=float)
            end = np.asarray(target["centroid"], dtype=float)
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#94a3b8", alpha=0.55, linewidth=1)

        for node in nodes:
            position = np.asarray(node["centroid"], dtype=float)
            label = str(node.get("label", "unknown"))
            color = palette(label_order[label] % 20)
            size = 24 + min(int(node.get("observations", 1)), 12) * 5
            ax.scatter(position[0], position[1], s=size, color=color, alpha=0.82, edgecolors="white", linewidths=0.4)

        target = str(nodes[0].get("metadata", {}).get("target", "unknown")) if nodes else "unknown"
        episode = graph_file.stem.removeprefix("episode_").removesuffix("_graph")
        top_labels = ", ".join(f"{label}×{count}" for label, count in labels.most_common(5))
        ax.set_title(f"Episode {episode} · target: {target}\n{len(nodes)} nodes / {len(graph.get('edges', []))} edges")
        ax.set_xlabel("world X (m)")
        ax.set_ylabel("world Y (m)")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.text(
            0.02,
            0.02,
            f"Top labels: {top_labels}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 3},
        )

    fig.suptitle("Current UAV-ON CLIP Scene Graphs · top-down world view", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_graphs(args.graph_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
