import json
from pathlib import Path
from typing import Any, Dict

from .graph import SceneGraph


def save_scene_graph(scene_graph: SceneGraph, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(scene_graph.to_dict(), file, indent=2, ensure_ascii=False)


def load_scene_graph(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)