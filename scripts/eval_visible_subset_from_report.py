"""Run the current UAV-ON scene-graph policy on strictly-visible episodes.

The visibility report is produced by ``find_strictly_visible_episode.py`` using
AirSim instance segmentation.  This wrapper keeps the evaluation subset honest:
only episodes whose target is actually rendered in the initial view are passed
to ``eval_simple_uavon.py``.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--visibility-report", required=True)
    parser.add_argument("--output-prefix", default="results/uavon/evaluation_strict_visible_batch")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--min-target-pixels", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = json.loads(Path(args.visibility_report).read_text())
    visible = [
        record for record in records
        if record.get("visible") and int(record.get("target_pixels") or 0) >= args.min_target_pixels
    ]
    visible.sort(key=lambda record: int(record.get("target_pixels") or 0), reverse=True)
    if args.max_episodes is not None:
        visible = visible[:args.max_episodes]
    if not visible:
        raise SystemExit("No strictly visible episodes matched the requested filters.")

    episode_ids = ",".join(str(record["episode_id"]) for record in visible)
    output_prefix = Path(args.output_prefix)
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_simple_uavon.py")),
        "--dataset", args.dataset,
        "--episode-ids", episode_ids,
        "--strategy", "hierarchical",
        "--detector", "groundingdino",
        "--clip-crop-verify",
        "--sam-segment",
        "--target-tiled-detection",
        "--target-clip-margin", "0.02",
        "--max-target-nodes-per-frame", "1",
        "--max-steps", str(args.max_steps),
        "--output", f"{output_prefix}.json",
        "--graph-dir", f"{output_prefix}_scene_graphs",
        "--save-frames-dir", f"{output_prefix}_frames",
        "--diagnostic-dir", f"{output_prefix}_diagnostics",
    ]

    print(f"Selected {len(visible)} strictly-visible episodes:")
    for record in visible:
        print(
            f"  episode={record['episode_id']} target={record.get('target_name')} "
            f"pixels={record.get('target_pixels')} bbox={record.get('bbox_xyxy')}"
        )
    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
