# ConceptGraphs to UAV-ON Migration

This repository now has a first lightweight ConceptGraphs-style module under `src/conceptgraphs_uav`.

Current scope:

- Reads UAV-ON RGB-D observations and AirSim poses.
- Projects depth maps into sparse world points with a pinhole camera model.
- Creates and merges 3D concept nodes by label and 3D bounding-box IoU.
- Adds simple spatial `near` edges and exports a JSON scene graph.

The first version avoids heavy dependencies such as Open3D, SAM, GroundingDINO, and CLIP so it can run in the existing UAV-ON environment. The open-vocabulary detection and CLIP feature fusion stages should be added behind the `ConceptGraphBuilder.add_frame()` boundary.

Basic module flow:

1. `UAVFrame.from_observation()` converts a UAV-ON observation into RGB, depth, pose, and metadata.
2. `ConceptGraphBuilder.add_frame()` projects depth into world coordinates and creates or updates a concept node.
3. `ConceptGraphBuilder.finalize()` adds spatial relations.
4. `save_scene_graph()` writes the graph to JSON.

Next integration step:

- Save `EvalBatchState.episodes[i]` after an evaluation rollout or call `ConceptGraphBuilder` online inside the evaluation loop.
- Replace frame-level labels with object proposals from GroundingDINO/SAM or another open-vocabulary detector.
- Store CLIP text/image embeddings per node for open-vocabulary querying.

Offline entry points:

- `PYTHONPATH=src python src/conceptgraphs_uav/build_from_episode.py --input episode.pkl --output graph.json`
- `PYTHONPATH=src python src/conceptgraphs_uav/build_from_trajectory.py --input logs/eval/.../trajectory.jsonl --output graph.json`

`build_from_trajectory.py` creates a pose-level graph from the existing UAV-ON evaluation logs. It is a bridge for current logs; the RGB-D ConceptGraph path should use `build_from_episode.py` or an online call to `ConceptGraphBuilder` once observations are saved.