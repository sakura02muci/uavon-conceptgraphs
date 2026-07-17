import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from conceptgraphs_uav.graph import ConceptGraphBuilder, ConceptNode, SceneGraph
from conceptgraphs_uav.io import save_scene_graph


def build_pose_graph_from_trajectory(input_path: str, output_path: str, label: str = "uav_pose") -> None:
    scene_graph = SceneGraph()
    previous_node_id = None
    for index, record in enumerate(_read_jsonl(input_path)):
        state = record["sensors"]["state"]
        position = [round(float(value), 4) for value in state["position"]]
        node_id = f"pose_{index:06d}"
        node = ConceptNode(
            node_id=node_id,
            label=label,
            caption=f"UAV pose at frame {record.get('frame', index)}",
            centroid=position,
            bbox_min=position,
            bbox_max=position,
            metadata={
                "frame": record.get("frame", index),
                "action": record.get("action"),
                "steps_size": record.get("steps_size"),
                "is_collision": record.get("is_collision", False),
                "distance_to_end": record.get("distance_to_end"),
            },
        )
        scene_graph.graph.add_node(node_id, **node.__dict__)
        if previous_node_id is not None:
            scene_graph.graph.add_edge(previous_node_id, node_id, relation="next")
        previous_node_id = node_id

    save_scene_graph(scene_graph, output_path)


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pose-level graph from a UAV-ON trajectory.jsonl file.")
    parser.add_argument("--input", required=True, help="Path to trajectory.jsonl")
    parser.add_argument("--output", required=True, help="Output scene graph JSON path")
    parser.add_argument("--label", default="uav_pose")
    args = parser.parse_args()
    build_pose_graph_from_trajectory(args.input, args.output, label=args.label)


if __name__ == "__main__":
    main()