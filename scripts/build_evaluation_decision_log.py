"""Export every saved ObjectNav decision into a compact Markdown log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def short(value: object, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def node_id(nav_target: object) -> str:
    if not isinstance(nav_target, dict):
        return "-"
    node = nav_target.get("target_node")
    if isinstance(node, dict):
        return str(node.get("node_id", "-"))
    planner = nav_target.get("planner")
    return str(planner.get("node_id", "-")) if isinstance(planner, dict) else "-"


def executor_status(nav_target: object) -> str:
    if not isinstance(nav_target, dict):
        return "-"
    executor = nav_target.get("executor")
    return str(executor.get("status", "-")) if isinstance(executor, dict) else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    episodes = json.loads(args.evaluation.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        lines = [
            f"# Episode {episode_id} · {episode['target']}：每步决策记录",
            "",
            f"结果：`{'成功' if episode.get('success') else '失败'}`；最小目标距离："
            f"`{float(episode.get('min_distance_to_goal', 0)):.2f} m`；"
            f"碰撞：`{episode.get('collision_count', 0)}` 次。",
            "",
            "| Step | 决策来源 | 动作 | 速度 | 子目标节点 | Executor 状态 | 动作前目标距离 | LLM/图决策 |",
            "|---:|---|---|---:|---|---|---:|---|",
        ]
        for record in episode.get("detections", []):
            nav_target = record.get("nav_target") or {}
            planner = nav_target.get("planner") if isinstance(nav_target, dict) else None
            reasoning = short(planner.get("reasoning")) if isinstance(planner, dict) else ""
            if isinstance(planner, dict) and planner.get("guard_forced"):
                reasoning = f"graph guard → {planner.get('guard_node_id')}; {reasoning}"
            distance = nav_target.get("uav_distance_to_true_goal_before_action") if isinstance(nav_target, dict) else None
            distance_text = f"{float(distance):.2f}" if distance is not None else "-"
            lines.append(
                f"| {int(record.get('step', 0))} | `{record.get('nav_source', '-')}` | "
                f"`{record.get('action', '-')}` | {float(record.get('action_speed') or 0):.2f} | "
                f"`{node_id(nav_target)}` | `{executor_status(nav_target)}` | {distance_text} m | {reasoning} |"
            )
        output = args.output_dir / f"episode_{episode_id}_decision_log.md"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
