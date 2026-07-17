"""Render per-step scene graph snapshots and the LLM-selected subgoal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def selected_point(nav_target: dict | None):
    if not isinstance(nav_target, dict):
        return None
    for key in ("subgoal", "centroid", "target_position"):
        if nav_target.get(key) is not None:
            return np.asarray(nav_target[key], dtype=float)
    executor = nav_target.get("executor")
    if isinstance(executor, dict) and executor.get("target_position") is not None:
        return np.asarray(executor["target_position"], dtype=float)
    return None


def render(snapshot_file: Path, output_file: Path) -> None:
    data = json.loads(snapshot_file.read_text(encoding="utf-8"))
    rgb_file = snapshot_file.with_name(snapshot_file.name.replace("step_", "rgb_").replace(".json", ".png"))
    rgb = np.asarray(Image.open(rgb_file).convert("RGB"))
    nodes = data["graph"].get("nodes", [])
    node_map = {str(node["node_id"]): node for node in nodes}

    fig, (image_ax, graph_ax) = plt.subplots(1, 2, figsize=(14, 6))
    image_ax.imshow(rgb)
    image_ax.set_title(f"RGB observation · step {data['step']}")
    image_ax.axis("off")

    for edge in data["graph"].get("edges", []):
        source, target = node_map.get(str(edge.get("source"))), node_map.get(str(edge.get("target")))
        if source is None or target is None:
            continue
        a, b = np.asarray(source["centroid"]), np.asarray(target["centroid"])
        graph_ax.plot([a[0], b[0]], [a[1], b[1]], color="#94a3b8", alpha=0.45, linewidth=1)

    for node in nodes:
        point = np.asarray(node["centroid"], dtype=float)
        label = str(node.get("label", "unknown"))
        is_target = data["target"].lower().replace(" ", "") in label.lower().replace(" ", "")
        graph_ax.scatter(point[0], point[1], s=55, c="#ef4444" if is_target else "#3b82f6", alpha=0.8)
        graph_ax.annotate(f"{node['node_id'].split('_')[-1]}:{label}", point[:2], fontsize=6, alpha=0.8)

    uav = np.asarray(data["uav_position"], dtype=float)
    goal = np.asarray(data["true_goal"], dtype=float)
    choice = selected_point(data.get("nav_target"))
    graph_ax.scatter(uav[0], uav[1], marker="^", s=130, c="black", label="UAV")
    graph_ax.scatter(goal[0], goal[1], marker="X", s=150, c="#22c55e", label="true goal")
    if choice is not None:
        graph_ax.scatter(choice[0], choice[1], marker="*", s=220, c="#a855f7", label="LLM subgoal")
        graph_ax.plot([uav[0], choice[0]], [uav[1], choice[1]], "--", color="#a855f7", linewidth=1.5)

    planner = (data.get("nav_target") or {}).get("planner", {}) if isinstance(data.get("nav_target"), dict) else {}
    chosen_id = planner.get("node_id") if isinstance(planner, dict) else None
    graph_ax.set_title(
        f"Graph after observation · {len(nodes)} nodes\n"
        f"source={data['nav_source']} · action={data['action']} · chosen={chosen_id}"
    )
    graph_ax.set_xlabel("world X (m)")
    graph_ax.set_ylabel("world Y (m)")
    graph_ax.grid(alpha=0.2)
    graph_ax.legend(loc="best", fontsize=8)
    graph_ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"Target: {data['target']} · UAV-to-goal: {data['distance_to_goal']:.2f} m")
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif", type=Path, default=None)
    args = parser.parse_args()
    rendered = []
    for snapshot in sorted(args.diagnostic_dir.glob("step_*.json")):
        output = args.output_dir / f"{snapshot.stem}.png"
        render(snapshot, output)
        rendered.append(output)
    if args.gif and rendered:
        frames = [Image.open(path).convert("RGB") for path in rendered]
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=700, loop=0)
    print(f"rendered={len(rendered)}")


if __name__ == "__main__":
    main()
