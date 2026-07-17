import argparse
import pickle
from pathlib import Path

from conceptgraphs_uav.frame import frames_from_episode
from conceptgraphs_uav.graph import ConceptGraphBuilder
from conceptgraphs_uav.io import save_scene_graph


def build_graph_from_episode_pickle(input_path: str, output_path: str, camera_index: int = 0) -> None:
    with Path(input_path).open("rb") as file:
        episode = pickle.load(file)

    builder = ConceptGraphBuilder()
    for frame in frames_from_episode(episode, camera_index=camera_index):
        builder.add_frame(frame)
    save_scene_graph(builder.finalize(), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight ConceptGraph from a saved UAV-ON episode pickle.")
    parser.add_argument("--input", required=True, help="Pickle file containing a list of UAV-ON observations")
    parser.add_argument("--output", required=True, help="Output scene graph JSON path")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()
    build_graph_from_episode_pickle(args.input, args.output, camera_index=args.camera_index)


if __name__ == "__main__":
    main()