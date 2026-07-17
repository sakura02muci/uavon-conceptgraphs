"""Closed-loop local waypoint executor for UAV-ON."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class LocalExecutorConfig:
    waypoint_tolerance: float = 1.5
    vertical_tolerance: float = 0.75
    turn_threshold_degrees: float = 18.0
    obstacle_distance: float = 2.5
    max_blocked_steps_before_ascend: int = 4
    slow_approach_radius: float = 8.0
    vertical_approach_radius: float = 4.0


class LocalWaypointExecutor:
    """Convert a world-space semantic subgoal into one safe control action."""

    def __init__(self, config: Optional[LocalExecutorConfig] = None) -> None:
        self.config = config or LocalExecutorConfig()
        self.blocked_steps = 0

    def reset(self) -> None:
        self.blocked_steps = 0

    def choose_action(
        self,
        position: np.ndarray,
        yaw: float,
        target_position: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[str, float, Dict[str, Any]]:
        position = np.asarray(position, dtype=np.float32)
        target_position = np.asarray(target_position, dtype=np.float32)
        planar_distance = float(np.linalg.norm(target_position[:2] - position[:2]))
        vertical_error = float(target_position[2] - position[2])
        distance_3d = float(np.linalg.norm(target_position - position))
        yaw_error = angle_error_to_point(position, target_position, yaw)
        clearance = depth_sector_clearance(depth)
        info: Dict[str, Any] = {
            "target_position": target_position.round(3).tolist(),
            "planar_distance": round(planar_distance, 3),
            "distance_3d": round(distance_3d, 3),
            "vertical_error": round(vertical_error, 3),
            "yaw_error_degrees": round(float(np.degrees(yaw_error)), 2),
            "clearance": {key: round(value, 3) for key, value in clearance.items()},
        }

        if distance_3d <= self.config.waypoint_tolerance:
            self.blocked_steps = 0
            return "hover", yaw_error, {**info, "status": "reached", "recommended_speed": 0.0}
        if planar_distance <= self.config.vertical_approach_radius:
            self.blocked_steps = 0
            if abs(vertical_error) <= 1.25 and planar_distance > self.config.waypoint_tolerance:
                if abs(np.degrees(yaw_error)) > self.config.turn_threshold_degrees:
                    return "turn_to_goal", yaw_error, {**info, "status": "final_xy_align", "recommended_speed": 0.0}
                speed = max(0.45, min(1.2, planar_distance * 0.35))
                return "forward_slow", yaw_error, {**info, "status": "final_xy_tracking", "recommended_speed": round(float(speed), 3)}
            if vertical_error > self.config.vertical_tolerance:
                speed = 1.5 if vertical_error > 4.0 else (1.0 if vertical_error > 2.0 else 0.6)
                return "descend_slow", 0.0, {**info, "status": "vertical_descend", "recommended_speed": speed}
            if vertical_error < -self.config.vertical_tolerance:
                speed = 1.5 if abs(vertical_error) > 4.0 else (1.0 if abs(vertical_error) > 2.0 else 0.6)
                return "ascend_slow", 0.0, {**info, "status": "vertical_ascend", "recommended_speed": speed}
            if planar_distance > self.config.waypoint_tolerance:
                if abs(np.degrees(yaw_error)) > self.config.turn_threshold_degrees:
                    return "turn_to_goal", yaw_error, {**info, "status": "final_xy_align", "recommended_speed": 0.0}
                speed = max(0.45, min(1.2, planar_distance * 0.35))
                return "forward_slow", yaw_error, {**info, "status": "final_xy_tracking", "recommended_speed": round(float(speed), 3)}
            return "hover", yaw_error, {**info, "status": "reached", "recommended_speed": 0.0}
        if abs(np.degrees(yaw_error)) > self.config.turn_threshold_degrees:
            return "turn_to_goal", yaw_error, {**info, "status": "aligning", "recommended_speed": 0.0}
        if np.isfinite(clearance["front"]) and clearance["front"] < self.config.obstacle_distance:
            self.blocked_steps += 1
            if self.blocked_steps >= self.config.max_blocked_steps_before_ascend:
                self.blocked_steps = 0
                return "ascend", 0.0, {**info, "status": "blocked_ascend", "recommended_speed": 2.0}
            turn_error = np.radians(30.0 if clearance["left"] >= clearance["right"] else -30.0)
            return "turn_to_goal", float(turn_error), {**info, "status": "avoiding", "recommended_speed": 0.0}
        self.blocked_steps = 0
        if planar_distance <= self.config.slow_approach_radius:
            speed = max(0.8, min(2.0, planar_distance * 0.45))
            action = "forward_slow"
        else:
            speed = max(2.0, min(4.0, planar_distance * 0.30))
            action = "forward"
        return action, yaw_error, {**info, "status": "tracking", "recommended_speed": round(float(speed), 3)}


def angle_error_to_point(position: np.ndarray, target_position: np.ndarray, yaw: float) -> float:
    direction = np.asarray(target_position)[:2] - np.asarray(position)[:2]
    target_angle = np.arctan2(direction[1], direction[0])
    return float((target_angle - yaw + np.pi) % (2 * np.pi) - np.pi)


def depth_sector_clearance(depth: np.ndarray) -> Dict[str, float]:
    """Estimate robust left/front/right free distance from a depth image."""
    if depth is None or np.asarray(depth).ndim != 2 or np.asarray(depth).size == 0:
        return {"left": float("inf"), "front": float("inf"), "right": float("inf")}
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    y1, y2 = int(height * 0.25), max(int(height * 0.78), int(height * 0.25) + 1)
    bounds = {
        "left": (int(width * 0.05), int(width * 0.38)),
        "front": (int(width * 0.40), int(width * 0.60)),
        "right": (int(width * 0.62), int(width * 0.95)),
    }
    result: Dict[str, float] = {}
    for name, (x1, x2) in bounds.items():
        values = depth[y1:y2, x1:x2]
        values = values[np.isfinite(values) & (values > 0.15)]
        result[name] = float(np.percentile(values, 25.0)) if values.size >= 24 else float("inf")
    return result
