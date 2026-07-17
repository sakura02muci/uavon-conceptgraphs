#!/usr/bin/env python3
"""Offline GroundingDINO detection test on saved UAV-ON RGB frames.

Runs entirely within the UAV_ON project layout and writes detections plus
annotated images under results/uavon/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "GroundingDINO"))

from conceptgraphs_uav.groundingdino_detector import GroundingDINODetector, draw_detections


def infer_target_from_episode(results_json: Path, episode_id: str) -> str | None:
    if not results_json.exists():
        return None
    with open(results_json, "r") as f:
        results = json.load(f)
    for item in results:
        if str(item.get("episode_id")) == str(episode_id):
            return item.get("target")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", default="results/uavon/frames_conceptgraph")
    parser.add_argument("--output-dir", default="results/uavon/groundingdino_frame_detections")
    parser.add_argument("--results-json", default="results/uavon/evaluation_conceptgraph.json")
    parser.add_argument("--checkpoint", default="checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--config", default=str(REPO_ROOT / "external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--episodes", nargs="*", default=None, help="Episode ids to process, e.g. 0 1 2")
    parser.add_argument("--max-frames-per-episode", type=int, default=8)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    frames_root = PROJECT_ROOT / args.frames_root
    output_dir = PROJECT_ROOT / args.output_dir
    results_json = PROJECT_ROOT / args.results_json
    checkpoint = PROJECT_ROOT / args.checkpoint

    detector = GroundingDINODetector(
        config_path=args.config,
        checkpoint_path=str(checkpoint),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    episode_dirs = sorted(p for p in frames_root.glob("episode_*") if p.is_dir())
    if args.episodes is not None:
        allowed = {str(e) for e in args.episodes}
        episode_dirs = [p for p in episode_dirs if p.name.split("episode_")[-1] in allowed]

    all_records = []
    for ep_dir in episode_dirs:
        episode_id = ep_dir.name.split("episode_")[-1]
        target = infer_target_from_episode(results_json, episode_id)
        rgb_files = sorted(ep_dir.glob("rgb_*.png"))[: args.max_frames_per_episode]
        print(f"\nEpisode {episode_id}: target={target}, frames={len(rgb_files)}")
        ep_records = []
        for rgb_path in rgb_files:
            image = np.asarray(Image.open(rgb_path).convert("RGB"))
            detections = detector.detect(image, target=target, top_k=args.top_k)
            step = rgb_path.stem.split("_")[-1]
            print(f"  {rgb_path.name}: {len(detections)} detections")
            for det in detections[:5]:
                print(f"    {det.label:18s} {det.confidence:.3f} {det.bbox_xyxy}")
            record = {
                "episode_id": episode_id,
                "target": target,
                "frame": rgb_path.name,
                "detections": [d.to_dict() for d in detections],
            }
            ep_records.append(record)
            all_records.append(record)
            draw_detections(image, detections, str(output_dir / f"episode_{episode_id}" / f"annotated_{step}.png"))

        with open(output_dir / f"episode_{episode_id}_detections.json", "w") as f:
            json.dump(ep_records, f, indent=2)

    with open(output_dir / "all_detections.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved detections and annotations to {output_dir}")


if __name__ == "__main__":
    main()
