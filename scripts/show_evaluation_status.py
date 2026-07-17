"""Print a compact summary for a resumable UAV-ON evaluation JSON file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    if not args.results.exists():
        print("No completed episode checkpoint yet.")
        return
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    completed = len(rows)
    successes = sum(bool(row.get("success")) for row in rows)
    mean_spl = sum(float(row.get("spl", 0.0)) for row in rows) / completed if completed else 0.0
    valid_distances = [
        float(value)
        for row in rows
        if (value := row.get("min_distance_to_goal")) is not None
    ]
    mean_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 0.0
    print(
        f"Completed: {completed}/{args.expected} | Success: {successes}/{completed} "
        f"| mean SPL: {mean_spl:.3f} | mean min-distance: {mean_distance:.2f} m"
    )
    for row in rows:
        raw_distance = row.get("min_distance_to_goal")
        distance_text = (
            f"{float(raw_distance):.2f} m"
            if raw_distance is not None
            else f"reset failed ({row.get('success_reason', 'unknown')})"
        )
        print(
            f"ep {str(row.get('episode_id')):>2} | {str(row.get('target')):<16} | "
            f"success={bool(row.get('success'))} | SPL={float(row.get('spl', 0.0)):.3f} | "
            f"min={distance_text}"
        )


if __name__ == "__main__":
    main()
