"""Build a Markdown trace with one RGB+graph panel and planner record per step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact_reasoning(value: object, limit: int = 420) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def planner_summary(nav_target: dict) -> str:
    planner = nav_target.get("planner") if isinstance(nav_target, dict) else None
    if not isinstance(planner, dict):
        return "无本轮 LLM 重规划（复用缓存子目标、局部执行器或碰撞恢复）。"
    parts = []
    original = planner.get("llm_original_node_id")
    final = planner.get("guard_node_id") or planner.get("node_id")
    if original and original != final:
        parts.append(f"LLM 原始节点：`{original}`；图守卫最终节点：`{final}`")
    elif final:
        parts.append(f"最终节点：`{final}`")
    if planner.get("guard_forced"):
        parts.append("图守卫为最终仲裁")
    if planner.get("direction"):
        parts.append(f"方向：`{planner['direction']}`")
    reasoning = compact_reasoning(planner.get("reasoning"))
    if reasoning:
        parts.append(f"理由：{reasoning}")
    return "； ".join(parts) if parts else "LLM 返回了空决策。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = json.loads(args.evaluation.read_text(encoding="utf-8"))
    episode = next((item for item in episodes if str(item.get("episode_id")) == str(args.episode_id)), None)
    if episode is None:
        raise ValueError(f"episode {args.episode_id} not found in {args.evaluation}")

    lines = [
        f"# Episode {episode['episode_id']} · {episode['target']}：逐步图像与决策轨迹",
        "",
        f"- 结果：`{'成功' if episode.get('success') else '失败'}`"
        f"；最小目标距离：`{float(episode['min_distance_to_goal']):.2f} m`"
        f"；碰撞：`{episode.get('collision_count', 0)}` 次。",
        "- 紫色星形是本轮最终执行的子目标；绿色 X 是评估时记录的真实目标，仅用于诊断，LLM 不可见。",
        "- `LLM 原始节点` 与 `图守卫最终节点` 不同，表示场景图的规则覆盖了 LLM 选择。",
        "",
    ]
    for record in episode.get("detections", []):
        step = int(record["step"])
        nav_target = record.get("nav_target") or {}
        collision = record.get("collision") or {}
        recovery = record.get("collision_recovery") or {}
        image = args.rendered_dir / f"step_{step:04d}.png"
        relative_image = image.relative_to(args.output.parent)
        lines.extend([
            f"## Step {step}",
            "",
            f"![step {step}]({relative_image.as_posix()})",
            "",
            f"- 执行：`{record.get('nav_source')}` → `{record.get('action')}`，速度 `{float(record.get('action_speed') or 0):.2f}`；动作后距真实目标 `{float(record.get('post_action_distance_to_goal') or 0):.2f} m`。",
            f"- 决策：{planner_summary(nav_target)}",
        ])
        if nav_target.get("near_target_refinement"):
            lines.append(
                f"- 近距视觉接管：连续观测 `{nav_target.get('visual_observations')}` 帧，"
                f"候选距离 `{nav_target.get('candidate_distance')} m`。"
            )
        if collision.get("is_new_collision"):
            lines.append(
                f"- 碰撞：`{collision.get('object_name')}`；恢复：`{recovery.get('reason')}`，"
                f"下一动作 `{recovery.get('recovery_action')}`。"
            )
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
