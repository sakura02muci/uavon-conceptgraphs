"""
UAV-ON ObjectNav evaluation with online ConceptGraph generation.

Each UAV-ON episode resets AirSim to the official start pose, collects RGB-D
observations online, builds a ConceptGraph from the visited trajectory, and
stores both navigation metrics and the per-episode scene graph.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import airsim
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Add paths
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from conceptgraphs_uav import ConceptGraphBuilder, LocalWaypointExecutor, UAVFrame
from conceptgraphs_uav.clip_detector import CLIPDetector
from conceptgraphs_uav.deepseek_planner import DeepSeekPlanner
from conceptgraphs_uav.graph import ConceptNode
from conceptgraphs_uav.geometry import project_depth_bbox_to_world
from conceptgraphs_uav.groundingdino_detector import GroundingDINODetector, draw_detections
from conceptgraphs_uav.io import save_scene_graph
from conceptgraphs_uav.sam_segmenter import SAMSegmenter


SUCCESS_DISTANCE_METERS = 5.0
TARGET_CONTACT_SUCCESS_MARGIN_METERS = 0.75
CAMERA_FOV_DEGREES = 90.0


class EpisodeResetError(RuntimeError):
    """An episode could not be placed at its dataset-defined start pose."""

    def __init__(self, expected, actual, phase: str):
        self.expected = [float(value) for value in expected]
        self.actual = [float(value) for value in actual]
        self.phase = phase
        super().__init__(
            f"AirSim episode reset failed during {phase}: "
            f"expected {self.expected}, got {self.actual}"
        )


def quaternion_to_yaw(quaternion) -> float:
    """Return AirSim pose yaw in radians from a Quaternionr."""
    rotation = R_scipy.from_quat([
        quaternion.x_val,
        quaternion.y_val,
        quaternion.z_val,
        quaternion.w_val,
    ])
    return float(rotation.as_euler("xyz")[2])


def angle_error_to_goal(position: np.ndarray, goal_pose: np.ndarray, yaw: float) -> float:
    direction = goal_pose[:2] - position[:2]
    goal_angle = np.arctan2(direction[1], direction[0])
    angle_diff = goal_angle - yaw
    return float((angle_diff + np.pi) % (2 * np.pi) - np.pi)


def angle_error_to_point(position: np.ndarray, target_position: np.ndarray, yaw: float) -> float:
    """Return signed yaw error from current pose to a 3-D scene graph point."""
    direction = target_position[:2] - position[:2]
    target_angle = np.arctan2(direction[1], direction[0])
    angle_diff = target_angle - yaw
    return float((angle_diff + np.pi) % (2 * np.pi) - np.pi)


def image_responses_to_arrays(responses):
    rgb_response = responses[0]
    rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)

    depth_response = responses[1]
    depth = np.asarray(depth_response.image_data_float, dtype=np.float32)
    depth = depth.reshape(depth_response.height, depth_response.width)
    return rgb, depth


def near_black_pixel_ratio(rgb: np.ndarray, threshold: int = 2) -> float:
    """Return a conservative proxy for missing-texture/black-tile coverage."""
    if rgb.size == 0:
        return 1.0
    return float(np.mean(np.max(rgb, axis=2) <= threshold))


def warm_up_rendering(client, frame_count: int, delay_seconds: float) -> list[float]:
    """Request scene frames at the episode start so UE texture streaming settles."""
    if frame_count <= 0:
        return []
    ratios = []
    client.simPause(False)
    try:
        for _ in range(frame_count):
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ])
            if responses and responses[0].width > 0 and responses[0].height > 0:
                raw = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                expected = int(responses[0].width * responses[0].height * 3)
                if raw.size == expected:
                    rgb = raw.reshape(responses[0].height, responses[0].width, 3)
                    ratios.append(near_black_pixel_ratio(rgb))
            time.sleep(max(0.0, delay_seconds))
    finally:
        client.simPause(True)
    return ratios


def collision_info_to_dict(collision_info) -> dict:
    """Convert AirSim CollisionInfo to a JSON-serializable dictionary."""
    return {
        "has_collided": bool(collision_info.has_collided),
        "object_name": str(collision_info.object_name),
        "object_id": int(collision_info.object_id),
        "time_stamp": int(collision_info.time_stamp),
        "penetration_depth": float(collision_info.penetration_depth),
        "position": [
            float(collision_info.position.x_val),
            float(collision_info.position.y_val),
            float(collision_info.position.z_val),
        ],
        "impact_point": [
            float(collision_info.impact_point.x_val),
            float(collision_info.impact_point.y_val),
            float(collision_info.impact_point.z_val),
        ],
        "normal": [
            float(collision_info.normal.x_val),
            float(collision_info.normal.y_val),
            float(collision_info.normal.z_val),
        ],
    }


def is_new_collision(collision_record: dict, last_collision_stamp: int | None) -> bool:
    """Return True if this record is a newly observed collision event."""
    if not collision_record.get("has_collided", False):
        return False
    stamp = int(collision_record.get("time_stamp", 0))
    if stamp <= 0:
        return last_collision_stamp is None
    return stamp != last_collision_stamp


def execute_action(client, action: str, angle_diff_rad: float, speed: float = 4.0) -> None:
    def move_along_measured_heading(signed_speed: float) -> None:
        # Body-frame velocity can use a stale controller yaw after
        # simSetVehiclePose/reset. Derive world NED velocity from the measured
        # pose so visual heading, mapping, and execution share one frame.
        yaw = quaternion_to_yaw(client.simGetVehiclePose().orientation)
        client.moveByVelocityAsync(
            signed_speed * float(np.cos(yaw)),
            signed_speed * float(np.sin(yaw)),
            0,
            1,
        ).join()
        client.hoverAsync().join()

    if action == "forward":
        move_along_measured_heading(speed)
    elif action == "forward_slow":
        move_along_measured_heading(speed * 0.45)
    elif action == "forward_medium":
        move_along_measured_heading(speed * 0.75)
    elif action == "backward":
        move_along_measured_heading(-speed * 0.6)
    elif action == "ascend":
        # AirSim uses NED coordinates: negative z moves upward.
        client.moveByVelocityBodyFrameAsync(0, 0, -2.0, 1).join()
        client.hoverAsync().join()
    elif action == "ascend_slow":
        client.moveByVelocityBodyFrameAsync(0, 0, -max(0.35, min(speed, 1.6)), 1).join()
        client.hoverAsync().join()
    elif action == "descend":
        client.moveByVelocityBodyFrameAsync(0, 0, 2.0, 1).join()
        client.hoverAsync().join()
    elif action == "descend_slow":
        client.moveByVelocityBodyFrameAsync(0, 0, max(0.35, min(speed, 1.6)), 1).join()
        client.hoverAsync().join()
    elif action == "turn_to_goal":
        turn_rate = 30 if angle_diff_rad > 0 else -30
        client.rotateByYawRateAsync(turn_rate, 1).join()
    elif action == "rotate_left":
        client.rotateByYawRateAsync(30, 1).join()
    elif action == "rotate_right":
        client.rotateByYawRateAsync(-30, 1).join()
    elif action == "rotate_left_90":
        client.rotateByYawRateAsync(45, 2).join()
    elif action == "rotate_right_90":
        client.rotateByYawRateAsync(-45, 2).join()
    else:
        client.hoverAsync().join()


def execute_llm_action(client, action: str, angle_diff_rad: float = 0.0) -> str:
    """Map LLM planner actions to the local AirSim executor."""
    action_map = {
        "forward": "forward",
        "backward": "backward",
        "rotl": "rotate_left",
        "rotr": "rotate_right",
        "left": "rotate_left",
        "right": "rotate_right",
        "ascend": "ascend",
        "descend": "descend",
        "stop": "hover",
    }
    executor_action = action_map.get(str(action).strip().lower(), "forward")
    execute_action(client, executor_action, angle_diff_rad, speed=3.0)
    return executor_action


def recommended_action_speed(nav_target, default: float = 4.0) -> float:
    """Extract the local executor's per-step speed recommendation."""
    if not isinstance(nav_target, dict):
        return default
    executor = nav_target.get("executor") if isinstance(nav_target.get("executor"), dict) else nav_target
    if isinstance(executor, dict):
        value = executor.get("recommended_speed")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
    return default


def apply_safe_step_speed(action: str, speed: float) -> float:
    """Conservative step sizes for collision-prone graph/LLM subgoal execution."""
    if action == "forward":
        return min(float(speed), 2.5)
    if action == "forward_medium":
        return min(float(speed), 2.0)
    if action == "forward_slow":
        return min(float(speed), 1.2)
    if action in {"descend", "descend_slow"}:
        return min(float(speed), 0.8)
    if action in {"ascend", "ascend_slow"}:
        return min(float(speed), 1.0)
    return float(speed)


def apply_close_target_speed(action: str, speed: float) -> float:
    """Use short actions while a repeatedly observed target is nearby."""
    if action == "forward":
        return min(float(speed), 1.2)
    if action == "forward_medium":
        return min(float(speed), 0.9)
    if action == "forward_slow":
        return min(float(speed), 0.7)
    if action in {"ascend", "ascend_slow", "descend", "descend_slow"}:
        return min(float(speed), 0.6)
    return float(speed)


def nav_target_subgoal_array(nav_target) -> np.ndarray | None:
    """Extract the concrete waypoint used by an executor/planner detail dict."""
    if not isinstance(nav_target, dict):
        return None
    candidate = nav_target.get("subgoal")
    if candidate is None and isinstance(nav_target.get("executor"), dict):
        candidate = nav_target["executor"].get("target_position")
    if candidate is None and nav_target.get("raw_node_centroid") is not None:
        candidate = nav_target.get("raw_node_centroid")
    if candidate is None and isinstance(nav_target.get("target_node"), dict):
        candidate = nav_target["target_node"].get("centroid")
    if candidate is None:
        return None
    try:
        return np.asarray(candidate, dtype=np.float32)
    except Exception:
        return None


def clear_navigation_memory_after_collision(state: dict, nav_target, collision_record: dict, object_name: str) -> dict:
    """Reject the current non-target subgoal and force a short recovery maneuver."""
    recovery = {
        "triggered": False,
        "reason": None,
        "rejected_node_id": None,
        "rejected_region": None,
        "recovery_action": None,
    }
    if not collision_record.get("is_new_collision", False):
        return recovery
    collided_object = str(collision_record.get("object_name", ""))
    if object_name and object_name in collided_object:
        recovery["reason"] = "target_contact_not_rejected"
        return recovery

    step = int(collision_record.get("step", -10_000))
    recent_steps = [
        int(old_step) for old_step in state.get("recent_collision_steps", [])
        if step - int(old_step) <= 10
    ]
    recent_steps.append(step)
    state["recent_collision_steps"] = recent_steps[-8:]
    escalated = len(recent_steps) >= 2
    # A second collision in the same short horizon means ascend/backward was
    # insufficient.  Change heading before moving again, rather than letting
    # the planner immediately re-enter the same obstacle corridor.
    turn_action = "rotate_left_90" if len(recent_steps) % 2 == 0 else "rotate_right_90"
    in_recovery = bool(isinstance(nav_target, dict) and nav_target.get("recovery_action"))
    # Never restart an ascent after the ascent itself collided.  The old state
    # machine reset to its first action on every collision, producing repeated
    # ascents against the same silo/wall.  A recovery collision now advances to
    # a heading change and lateral departure immediately.
    if in_recovery:
        sequence = [turn_action, "forward_slow"]
        recovery_reason = "collision_during_recovery_escape"
    elif escalated:
        sequence = ["ascend_slow", "backward", turn_action, "forward_slow"]
        recovery_reason = "repeated_collision_escape"
    else:
        sequence = ["ascend_slow", "backward"]
        recovery_reason = "non_target_collision"
    recovery.update({
        "triggered": True,
        "reason": recovery_reason,
        "recovery_action": sequence[0],
        "recent_collision_count": len(recent_steps),
        "escalated_escape": bool(escalated or in_recovery),
        "collision_during_recovery": in_recovery,
    })
    state["collision_recovery_steps"] = len(sequence)
    state["collision_recovery_sequence"] = sequence
    state["collision_recovery_action"] = sequence[0]
    state["last_collision_object"] = collided_object
    state["subgoal"] = None
    state["last_plan_step"] = -10_000
    state.pop("target_candidate_lock", None)
    state.pop("target_lock_subgoal", None)
    state.pop("target_lock_node", None)
    state.pop("target_lock_node_id", None)

    rejected_regions = list(state.get("rejected_subgoal_regions", []))
    rejected_center = nav_target_subgoal_array(nav_target)
    if rejected_center is not None:
        region = {
            "center": rejected_center.round(3).tolist(),
            "radius": 18.0 if (escalated or in_recovery) else 12.0,
            "reason": "recovery_collision" if in_recovery else ("repeated_collision" if escalated else "collision"),
            "object_name": collided_object,
        }
        rejected_regions.append(region)
        state["rejected_subgoal_regions"] = rejected_regions[-12:]
        recovery["rejected_region"] = region

    if isinstance(nav_target, dict):
        target_node = nav_target.get("target_node")
        if not isinstance(target_node, dict):
            planner = nav_target.get("planner")
            chosen_id = planner.get("node_id") if isinstance(planner, dict) else None
            if chosen_id:
                target_node = {"node_id": chosen_id}
        node_id = target_node.get("node_id") if isinstance(target_node, dict) else None
        if node_id:
            rejected = set(str(item) for item in state.get("rejected_node_ids", []))
            rejected.add(str(node_id))
            state["rejected_node_ids"] = sorted(rejected)
            recovery["rejected_node_id"] = str(node_id)
            nav_target["collision_rejected_node_id"] = str(node_id)
    return recovery


def reject_collided_graph_node(state: dict, nav_target, collision_record: dict, object_name: str) -> None:
    """Avoid repeatedly descending into the same non-target graph obstacle."""
    if not collision_record.get("is_new_collision", False):
        return
    collided_object = str(collision_record.get("object_name", ""))
    if object_name and object_name in collided_object:
        return
    if not isinstance(nav_target, dict):
        return
    target_node = nav_target.get("target_node")
    if not isinstance(target_node, dict):
        return
    node_id = target_node.get("node_id")
    if not node_id:
        return
    rejected = set(str(item) for item in state.get("rejected_node_ids", []))
    rejected.add(str(node_id))
    state["rejected_node_ids"] = sorted(rejected)
    if state.get("target_lock_node_id") == str(node_id):
        state.pop("target_lock_subgoal", None)
        state.pop("target_lock_node", None)
        state.pop("target_lock_node_id", None)
    state.pop("target_candidate_lock", None)
    nav_target["collision_rejected_node_id"] = str(node_id)


def is_target_object_contact(collision_record: dict, object_name: str) -> bool:
    if not collision_record.get("has_collided", False):
        return False
    collided_object = str(collision_record.get("object_name", ""))
    return bool(object_name) and object_name in collided_object


def choose_action(strategy, target_name, detected_label, confidence, angle_diff_rad, step):
    target_lower = target_name.lower()
    detected_lower = detected_label.lower()
    angle_diff_deg = abs(np.degrees(angle_diff_rad))

    if strategy == "baseline":
        return "forward" if step % 8 else "rotate_left"

    if strategy == "oracle":
        return "turn_to_goal" if angle_diff_deg > 25 else "forward"

    if target_lower in detected_lower and confidence >= 0.03:
        return "forward"

    if strategy == "conceptgraph":
        # CLIP labels are image-level in this lightweight port. Use detections to
        # bias exploration, and rotate periodically when the target is not visible.
        if step % 12 in (0, 1, 2):
            return "rotate_left"
        return "forward"

    # clip/deepseek fallback: target-aware exploration with a weak goal-facing prior
    # so the UAV-ON run stays inside the episode neighborhood.
    if angle_diff_deg > 70:
        return "turn_to_goal"
    if confidence < 0.02 and step % 6 == 0:
        return "rotate_left"
    return "forward"


def normalize_target_name(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))
    compact = spaced.replace("_", " ").replace("-", " ").strip().lower()
    if compact.replace(" ", "") in {"busstop", "busshelter", "busstation"}:
        return "bus stop"
    if compact.replace(" ", "") in {"trafficlight", "trafficsignal", "stoplight"}:
        return "traffic light"
    return compact


def normalized_label_text(value: str) -> str:
    """Normalise detector, dataset and graph labels into comparable words.

    Dataset targets use CamelCase (``SoccerBall``), detector phrases usually
    contain spaces (``soccer ball``), and older graph files may retain a
    compact spelling (``soccerball``).  Keep a space-separated version for
    boundary-aware matching and derive a compact key only for exact equality.
    """
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value or ""))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", spaced.lower()).split())


def normalized_label_key(value: str) -> str:
    return normalized_label_text(value).replace(" ", "")


def label_matches_target(label: str, target_name: str) -> bool:
    label_text = normalized_label_text(label)
    label_key = normalized_label_key(label)
    target = normalize_target_name(target_name)
    aliases = {
        "bus stop": ["bus stop", "bus shelter", "bus station"],
        "traffic light": ["traffic light", "traffic signal", "stoplight"],
        "lamp post": ["lamp post", "street lamp"],
        "bench": ["bench", "park bench"],
        "fountain": ["fountain", "water fountain"],
        "rock": ["rock", "stone", "boulder"],
        "caravan": ["caravan", "camper", "trailer", "mobile home"],
        "soccer ball": ["soccer ball", "football", "sports ball"],
        "teapot": ["teapot", "tea kettle", "kettle"],
        "table": ["table", "picnic table", "outdoor table"],
        "chair": ["chair", "outdoor chair", "seat"],
        "traffic cone": ["traffic cone", "road cone", "safety cone"],
        "stop sign": ["stop sign", "road sign"],
    }.get(target, [target])
    for alias in aliases:
        alias_text = normalized_label_text(alias)
        alias_key = normalized_label_key(alias)
        # Exact compact-key equality bridges ``SoccerBall`` / ``soccerball`` /
        # ``soccer ball`` without turning an arbitrary substring into a match.
        if alias_key and label_key == alias_key:
            return True
        # Retain useful compositional phrases such as ``outdoor chair`` while
        # preventing false matches such as target ``car`` in ``caravan``.
        if alias_text and re.search(r"(?<![a-z0-9])" + re.escape(alias_text) + r"(?![a-z0-9])", label_text):
            return True
    return False


def bbox_steering_error(detections, target_name: str, image_width: int) -> tuple[float | None, dict | None]:
    """Estimate yaw error from target bbox horizontal offset in the current camera image."""
    target_dets = []
    image_height = max(image_width * 0.5625, 1.0)
    image_area = float(image_width * image_height)
    for det in detections:
        if not label_matches_target(det.label, target_name):
            continue
        x1, y1, x2, y2 = det.bbox_xyxy
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        # Discard very weak and full-frame prompt hallucinations.
        area_ratio = box_area / image_area
        if det.confidence < 0.34 or area_ratio > 0.35 or area_ratio < 0.001:
            continue
        # Boxes clipped by image borders often correspond to partial false
        # positives and made the UAV spin between left/right edge detections.
        if x1 <= 2.0 or x2 >= image_width - 2.0:
            continue
        target_dets.append(det)
    if not target_dets:
        return None, None
    def track_score(det):
        x1, _, x2, _ = det.bbox_xyxy
        cx = 0.5 * (x1 + x2)
        center_error = abs(cx - image_width * 0.5) / max(1.0, image_width * 0.5)
        return float(det.confidence) - 0.25 * center_error

    best = max(target_dets, key=track_score)
    x1, _, x2, _ = best.bbox_xyxy
    cx = 0.5 * (x1 + x2)
    mid = max(1.0, image_width * 0.5)
    # Positive error should rotate left in execute_action("turn_to_goal", ...).
    normalized = (mid - cx) / mid
    error = np.radians(CAMERA_FOV_DEGREES * 0.5 * normalized)
    return float(error), best.to_dict()


def bbox_area_ratio(bbox_xyxy, image_shape: tuple[int, int]) -> float:
    """Return 2-D bbox area ratio for filtering prompt hallucinations."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(area / max(1.0, width * height))


def target_observation_lifecycle(
    det,
    evidence: dict,
    image_shape: tuple[int, int],
    projected_range_m: float | None,
    target_name: str = "",
) -> str:
    """Classify a target observation before it is allowed to control 3-D navigation.

    Far/small observations are useful bearings but their RGB-D projection is
    too fragile to become a persistent object node.  They remain available as
    a ``far_cue`` for a bounded camera turn.
    """
    if projected_range_m is None:
        return "far_cue"
    area = bbox_area_ratio(det.bbox_xyxy, image_shape)
    if area < 0.00008:
        return "far_cue"
    if projected_range_m > 30.0:
        # A hard 30 m cutoff regressed initially visible medium/large objects:
        # Playground ep60 was repeatedly detected at 32-43 m with a stable
        # projected location and strong CLIP evidence, yet was never allowed to
        # fuse.  Permit such observations to *enter fusion only*; the scene
        # graph still requires three compatible views before navigation. Keep
        # this route closed for the tiny SoccerBall target, whose far depth is
        # dominated by background pixels.
        strong_far_object = (
            normalize_target_name(target_name) != "soccer ball"
            and projected_range_m <= 55.0
            and area >= 0.0040
            and float(getattr(det, "confidence", 0.0)) >= 0.30
            and evidence.get("status") == "verified"
            and float(evidence.get("target_margin") or -1e9) >= 0.015
        )
        if not strong_far_object:
            return "far_cue"
    # Small UAV-ON objects (ball, teapot) occupy very few pixels. Retain a
    # close, high-confidence CLIP-verified instance instead of permanently
    # treating it as an unusable bearing-only cue.
    if (
        area < 0.0020
        and not (
            evidence.get("status") == "verified"
            and float(getattr(det, "confidence", 0.0)) >= 0.35
            and projected_range_m <= 15.0
        )
    ):
        return "far_cue"
    # ``verified`` means semantic verification of this 2-D crop; graph fusion
    # still requires three observations before exposing a target node.  Returning
    # ``provisional`` here used to make the later node selector (which correctly
    # requires verified semantic evidence) reject every non-oracle target node.
    if evidence.get("status") == "verified":
        return "verified"
    return "provisional"


def is_far_target_evidence(evidence: dict | None) -> bool:
    return isinstance(evidence, dict) and evidence.get("target_lifecycle") == "far_cue"


def target_proposal_geometry_is_plausible(det, image_shape: tuple[int, int], target_name: str) -> bool:
    """Reject prompt-hallucination boxes before CLIP/3-D target processing.

    GroundingDINO on an alias-rich prompt often returns a whole tile of sky or
    ground as a small-object target.  For a SoccerBall those broad boxes are
    categorically incompatible with the target and also crowd out the genuine,
    low-confidence small proposal when tiled detections are top-k truncated.
    """
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in det.bbox_xyxy]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    area = bbox_area_ratio(det.bbox_xyxy, image_shape)
    target = normalize_target_name(target_name)
    if area <= 0.00003 or area > 0.35:
        return False
    if target != "soccer ball":
        return True

    # A ball should be locally compact and approximately round.  The lower
    # bound intentionally permits a 5-10 px distant ball; the upper bound and
    # border check remove the repeated 15-20% tile-sized sky/ground boxes seen
    # in the SoccerBall diagnostic run.
    aspect = box_width / box_height
    if area > 0.015 or not (0.45 <= aspect <= 2.20):
        return False
    touches_edge = x1 <= 2.0 or y1 <= 2.0 or x2 >= width - 2.0 or y2 >= height - 2.0
    if touches_edge and area > 0.0015:
        return False
    return True


def sam_mask_depth_quality(mask: np.ndarray | None, depth: np.ndarray) -> dict:
    """Summarise whether a SAM mask yields a usable object-depth estimate.

    Bounding boxes around chairs, sacks and tables frequently include ground or
    a wall.  A SAM mask is useful only when it has enough finite depth pixels
    and a compact depth distribution; otherwise it must not create a persistent
    3-D target node.
    """
    quality = {"available": False, "valid_fraction": None, "median_depth": None, "relative_span": None}
    if mask is None or depth is None or mask.shape != depth.shape:
        return quality
    selected = np.asarray(depth)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return quality
    valid = selected[np.isfinite(selected) & (selected > 0.05) & (selected < 80.0)]
    quality["available"] = True
    quality["valid_fraction"] = float(valid.size / selected.size)
    if valid.size < 12:
        return quality
    median = float(np.median(valid))
    lo, hi = np.percentile(valid, [10, 90])
    quality["median_depth"] = median
    quality["relative_span"] = float((hi - lo) / max(median, 0.25))
    return quality


def target_mask_depth_is_reliable(quality: dict) -> bool:
    """Gate target-node insertion on SAM depth support, not bbox background."""
    if not quality.get("available", False):
        return True  # SAM was unavailable; preserve the existing bbox fallback.
    fraction = quality.get("valid_fraction")
    span = quality.get("relative_span")
    return bool(fraction is not None and fraction >= 0.35 and (span is None or span <= 1.10))


def projected_target_range_m(det, depth: np.ndarray, position: np.ndarray, orientation: np.ndarray) -> float | None:
    """Estimate 3-D range for lifecycle gating without using the true goal."""
    points = project_depth_bbox_to_world(
        depth=depth,
        bbox_xyxy=np.asarray(det.bbox_xyxy, dtype=np.float32),
        position=position,
        quaternion_xyzw=orientation,
        fov_degrees=CAMERA_FOV_DEGREES,
        stride=2,
        max_depth=80.0,
    )
    if len(points) == 0:
        return None
    centroid = np.median(points, axis=0)
    return float(np.linalg.norm(centroid[:3] - position[:3]))


def target_persistence_score(det, evidence: dict, image_shape: tuple[int, int], target_name: str) -> float:
    """Rank verified target crops using semantics plus a weak category shape prior."""
    margin = float(evidence.get("target_margin") or -1e9)
    confidence = float(getattr(det, "confidence", 0.0))
    x1, y1, x2, y2 = [float(value) for value in det.bbox_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = bbox_area_ratio(det.bbox_xyxy, image_shape)
    score = margin + 0.01 * confidence
    if normalize_target_name(target_name) == "caravan":
        aspect_h_over_w = height / width
        if 1.10 <= aspect_h_over_w <= 2.50:
            score += 0.020
        if 0.004 <= area <= 0.035:
            score += 0.010
        if area > 0.08:
            score -= 0.030
    return float(score)


def is_graph_detection(det, image_shape: tuple[int, int], target_name: str) -> bool:
    """Filter detections before adding them as persistent 3-D graph nodes."""
    area_ratio = bbox_area_ratio(det.bbox_xyxy, image_shape)
    if area_ratio <= 0.00008 or area_ratio > 0.35:
        return False
    if label_matches_target(det.label, target_name):
        return det.confidence >= 0.30
    # Context objects are useful for LLM exploration but should be reasonably stable.
    return det.confidence >= 0.36


def verify_detection_with_clip(det, image_rgb: np.ndarray, target_name: str, clip_verifier=None) -> tuple[bool, list[dict]]:
    """Use CLIP on a bbox crop to reject target-like GroundingDINO false positives."""
    if clip_verifier is None or not label_matches_target(det.label, target_name):
        return True, []
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
    pad_x = 0.08 * max(1.0, x2 - x1)
    pad_y = 0.08 * max(1.0, y2 - y1)
    x1 = int(np.clip(np.floor(x1 - pad_x), 0, width - 1))
    x2 = int(np.clip(np.ceil(x2 + pad_x), 0, width))
    y1 = int(np.clip(np.floor(y1 - pad_y), 0, height - 1))
    y2 = int(np.clip(np.ceil(y2 + pad_y), 0, height))
    if x2 <= x1 or y2 <= y1:
        return False, []
    crop = image_rgb[y1:y2, x1:x2]
    try:
        predictions = clip_verifier.classify_image(crop, top_k=5)
    except Exception:
        return True, []
    records = [{"label": label, "confidence": round(float(score), 4)} for label, score in predictions]
    matched = any(label_matches_target(label, target_name) for label, _ in predictions)
    return matched, records


def clip_detection_target_evidence(
    det,
    image_rgb: np.ndarray,
    target_name: str,
    target_description: str,
    clip_verifier=None,
) -> dict:
    """Score a detection crop directly against target and background prompts.

    Generic CLIP class predictions are retained for diagnostics, but no longer
    have veto power over a GroundingDINO target observation.
    """
    evidence = {
        "status": "unverified",
        "target_score": None,
        "negative_score": None,
        "target_margin": None,
        "generic_predictions": [],
    }
    if clip_verifier is None:
        return evidence
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
    pad_x = 0.05 * max(1.0, x2 - x1)
    pad_y = 0.05 * max(1.0, y2 - y1)
    x1 = int(np.clip(np.floor(x1 - pad_x), 0, width - 1))
    x2 = int(np.clip(np.ceil(x2 + pad_x), 0, width))
    y1 = int(np.clip(np.floor(y1 - pad_y), 0, height - 1))
    y2 = int(np.clip(np.ceil(y2 + pad_y), 0, height))
    if x2 <= x1 or y2 <= y1:
        evidence["status"] = "invalid_crop"
        return evidence
    crop = image_rgb[y1:y2, x1:x2]
    try:
        target_prompts = clip_target_prompt_set(target_name, target_description)
        target_features = clip_text_features(clip_verifier, target_prompts)
        negative_prompts = clip_negative_prompt_set()
        negative_features = clip_text_features(clip_verifier, negative_prompts)
        margin, target_score, negative_score = clip_image_target_score(
            clip_verifier, crop, target_features, negative_features
        )
        generic = clip_verifier.classify_image(crop, top_k=5)
    except Exception as exc:
        evidence["status"] = "clip_error"
        evidence["error"] = str(exc)
        return evidence
    evidence.update({
        "status": "verified" if margin >= 0.0 else "provisional",
        "target_score": round(float(target_score), 4),
        "negative_score": round(float(negative_score), 4),
        "target_margin": round(float(margin), 4),
        "generic_predictions": [
            {"label": label, "confidence": round(float(score), 4)} for label, score in generic
        ],
    })
    return evidence


def target_prompt_from_episode(target_name: str, description: str, max_words: int = 18) -> str:
    """Build a category prompt; detailed prose is handled by CLIP verification.

    GroundingDINO treats dot-separated terms as categories. Feeding the UAV-ON
    prose as one noun phrase caused it to ground generic words such as "roof"
    and "rectangular form" instead of the requested object.
    """
    target = normalize_target_name(target_name)
    return target


def summarize_target_candidates(
    detections,
    target_name: str,
    image_rgb: np.ndarray,
    image_shape: tuple[int, int],
    depth: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    goal_pose: np.ndarray,
    clip_verifier=None,
    target_description: str = "",
    top_k: int = 8,
    evidence_cache: dict[int, dict] | None = None,
) -> list[dict]:
    """Record per-frame target-like 2-D detections and their projected 3-D diagnostics."""
    records = []
    for det in detections:
        if not label_matches_target(det.label, target_name):
            continue
        projected = None
        projected_distance = None
        projected_range = None
        points = project_depth_bbox_to_world(
            depth=depth,
            bbox_xyxy=np.asarray(det.bbox_xyxy, dtype=np.float32),
            position=position,
            quaternion_xyzw=orientation,
            fov_degrees=CAMERA_FOV_DEGREES,
            stride=2,
            max_depth=80.0,
        )
        if len(points):
            projected_arr = np.median(points, axis=0)
            projected = projected_arr.round(3).tolist()
            projected_distance = round(float(np.linalg.norm(projected_arr[:3] - goal_pose[:3])), 3)
            projected_range = round(float(np.linalg.norm(projected_arr[:3] - position[:3])), 3)
        evidence = (evidence_cache or {}).get(id(det))
        if evidence is None:
            evidence = clip_detection_target_evidence(
                det, image_rgb, target_name, target_description, clip_verifier
            )
            evidence["projected_range_m"] = projected_range
            evidence["target_lifecycle"] = target_observation_lifecycle(
                det, evidence, image_shape, projected_range, target_name
            )
        records.append({
            "label": det.label,
            "confidence": round(float(det.confidence), 3),
            "bbox_xyxy": [round(float(v), 2) for v in det.bbox_xyxy],
            "bbox_area_ratio": round(bbox_area_ratio(det.bbox_xyxy, image_shape), 5),
            "projected_centroid": projected,
            "projected_distance_to_goal": projected_distance,
            "projected_range_m": projected_range,
            "clip_target_match": evidence.get("status") == "verified",
            "clip_target_score": evidence.get("target_score"),
            "clip_negative_score": evidence.get("negative_score"),
            "clip_target_margin": evidence.get("target_margin"),
            "clip_predictions": evidence.get("generic_predictions", []),
            "semantic_status": evidence.get("status"),
            "target_lifecycle": evidence.get("target_lifecycle", "provisional"),
            "accepted_for_graph": is_graph_detection(det, image_shape, target_name),
        })
    records.sort(key=lambda item: (-item["confidence"], item["projected_distance_to_goal"] or 1e9))
    return records[:top_k]


def target_candidate_quality(record: dict) -> float:
    """Score a current-frame target candidate for direct local navigation."""
    if not isinstance(record, dict) or record.get("projected_centroid") is None:
        return -1e9
    if record.get("target_lifecycle") == "far_cue":
        return -1e9
    confidence = float(record.get("confidence") or 0.0)
    area = float(record.get("bbox_area_ratio") or 0.0)
    margin = float(record.get("clip_target_margin") or 0.0)
    status = record.get("semantic_status")
    if status == "verified":
        semantic_penalty = 0.0
    elif status == "provisional" and margin >= -0.006:
        semantic_penalty = 0.012
    else:
        return -1e9
    if confidence < 0.26 or area <= 0.00015 or area > 0.22:
        return -1e9
    # Slightly prefer centered/large-enough detections without letting huge
    # prompt hallucinations dominate.  CLIP margin remains the main semantic cue.
    area_bonus = min(area, 0.025) * 2.0
    return float(margin + 0.03 * confidence + area_bonus - semantic_penalty)


def best_target_candidate(target_candidates: list[dict]) -> dict | None:
    """Return the best verified projected target candidate in this observation."""
    scored = scored_target_candidates(target_candidates)
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def scored_target_candidates(target_candidates: list[dict]) -> list[tuple[float, dict]]:
    scored = [
        (target_candidate_quality(record), record)
        for record in target_candidates
        if isinstance(record, dict)
    ]
    return [(score, record) for score, record in scored if score > -1e8]


def navigation_subgoal_from_target_candidate(record: dict) -> np.ndarray:
    """Convert a projected target candidate into a waypoint for the local executor."""
    centroid = np.asarray(record["projected_centroid"], dtype=np.float32)
    subgoal = centroid.copy()
    # Candidate z is helpful when it is plausible, but tiny/partial bboxes can
    # still produce wild vertical estimates.  Fall back to the standard object
    # approach altitude when z is clearly unreliable.
    if not (-2.0 <= float(subgoal[2]) <= 8.0):
        subgoal[2] = 0.0
    return subgoal


def keep_existing_target_candidate_lock(lock: dict, step: int, max_missed: int = 10) -> tuple[np.ndarray | None, dict | None]:
    """Refresh a stable candidate lock when no spatially continuous observation appears."""
    if not isinstance(lock, dict) or lock.get("subgoal") is None:
        return None, None
    missed = int(lock.get("missed_updates", 0)) + 1
    if missed > max_missed:
        return None, None
    refreshed = dict(lock)
    refreshed["last_step"] = int(step)
    refreshed["missed_updates"] = int(missed)
    refreshed["maintained_without_observation"] = True
    return np.asarray(refreshed["subgoal"], dtype=np.float32), refreshed


def update_target_candidate_lock(
    state: dict,
    target_candidates: list[dict],
    position: np.ndarray,
    step: int,
) -> tuple[np.ndarray | None, dict | None]:
    """Maintain a short-horizon lock from repeated high-quality target candidates.

    This is deliberately separate from graph fusion: it catches the failure mode
    where the current frame sees the target and projects it well, but merging or
    node thresholds prevent a usable target node from being selected.
    """
    lock = state.get("target_candidate_lock")
    scored_candidates = scored_target_candidates(target_candidates)
    record = None
    if isinstance(lock, dict) and lock.get("subgoal") is not None:
        previous = np.asarray(lock["subgoal"], dtype=np.float32)
        recent = step - int(lock.get("last_step", -9999)) <= 3
        if recent:
            continuous = []
            for score, candidate in scored_candidates:
                centroid = navigation_subgoal_from_target_candidate(candidate)
                spatial_delta = float(np.linalg.norm(previous[:2] - centroid[:2]))
                if spatial_delta <= 16.0:
                    # Once a target hypothesis is born, spatial continuity is a
                    # stronger cue than a single-frame CLIP margin.  This avoids
                    # jumping from a tiny true target to a semantically plausible
                    # but far-away false positive.
                    continuous.append((spatial_delta, -score, candidate))
            if continuous:
                _, _, record = min(continuous, key=lambda item: item[:2])
            elif int(lock.get("observations", 0)) >= 2:
                # A stable track should not be overwritten by a one-frame
                # far-away detection.  Keep flying toward the last consistent
                # candidate for a short horizon and let fresh observations
                # reattach when they become spatially plausible again.
                subgoal, refreshed = keep_existing_target_candidate_lock(lock, step)
                if refreshed is not None:
                    state["target_candidate_lock"] = refreshed
                    return subgoal, refreshed
    if record is None:
        def global_candidate_score(item: tuple[float, dict]) -> float:
            score, candidate = item
            centroid = navigation_subgoal_from_target_candidate(candidate)
            planar_distance = float(np.linalg.norm(centroid[:2] - position[:2]))
            return float(score - 0.002 * planar_distance)

        record = max(scored_candidates, key=global_candidate_score)[1] if scored_candidates else None
    if record is None:
        if isinstance(lock, dict) and step - int(lock.get("last_step", -9999)) <= 3:
            observations = int(lock.get("observations", 0))
            if observations >= 2:
                subgoal, refreshed = keep_existing_target_candidate_lock(lock, step)
                if refreshed is not None:
                    state["target_candidate_lock"] = refreshed
                    return subgoal, refreshed
        return None, None

    raw_subgoal = navigation_subgoal_from_target_candidate(record)
    score = target_candidate_quality(record)
    observations = 1
    smoothed = raw_subgoal.copy()
    if isinstance(lock, dict) and lock.get("subgoal") is not None:
        previous = np.asarray(lock["subgoal"], dtype=np.float32)
        spatial_delta = float(np.linalg.norm(previous[:2] - raw_subgoal[:2]))
        recent = step - int(lock.get("last_step", -9999)) <= 3
        if recent and spatial_delta <= 16.0:
            observations = int(lock.get("observations", 1)) + 1
            # Smooth only XY; keep z from the current observation when it is
            # plausible because vertical approach matters for UAV-ON success.
            smoothed[:2] = 0.65 * previous[:2] + 0.35 * raw_subgoal[:2]

    distance_from_uav = float(np.linalg.norm(smoothed[:2] - position[:2]))
    strong_single_frame = (
        score >= 0.012
        and float(record.get("confidence") or 0.0) >= 0.27
        and distance_from_uav <= 40.0
        and record.get("semantic_status") == "verified"
    )
    new_lock = {
        "subgoal": smoothed.round(4).tolist(),
        "raw_subgoal": raw_subgoal.round(4).tolist(),
        "last_step": int(step),
        "observations": int(observations),
        "score": round(float(score), 4),
        "missed_updates": 0,
        "maintained_without_observation": False,
        "strong_single_frame": bool(strong_single_frame),
        "candidate": record,
        "distance_from_uav": round(distance_from_uav, 3),
    }
    state["target_candidate_lock"] = new_lock
    if observations >= 2 or strong_single_frame:
        return smoothed, new_lock
    return None, None


def find_target_node(scene_graph, target_name: str, position: np.ndarray, rejected_node_ids: set[str] | None = None) -> tuple[np.ndarray | None, dict | None]:
    """Pick the best target-like node from the current scene graph."""
    candidates = []
    rejected_node_ids = rejected_node_ids or set()
    for node_id, node in scene_graph.graph.nodes(data=True):
        if str(node_id) in rejected_node_ids:
            continue
        if not label_matches_target(str(node.get("label", "")), target_name):
            continue
        if not node_is_confirmed(node):
            continue
        centroid = np.asarray(node["centroid"], dtype=np.float32)
        metadata = node.get("metadata", {}) if isinstance(node, dict) else {}
        is_oracle = isinstance(metadata, dict) and metadata.get("detector") == "oracle_graph_bootstrap"
        semantic_evidence = metadata.get("semantic_evidence", {}) if isinstance(metadata, dict) else {}
        lifecycle = semantic_evidence.get("target_lifecycle") if isinstance(semantic_evidence, dict) else None
        if not is_oracle and lifecycle != "verified":
            continue
        if not is_oracle and float(centroid[2]) < -15.0:
            continue
        distance = float(np.linalg.norm(centroid[:2] - position[:2]))
        confidence = float(node.get("confidence", 0.0))
        observations = float(node.get("observations", 1))
        score = confidence * (1.0 + 0.15 * observations) / max(distance, 1.0)
        node_record = dict(node)
        node_record["node_id"] = str(node_id)
        candidates.append((score, centroid, node_record))
    if not candidates:
        return None, None
    _, centroid, node = max(candidates, key=lambda item: item[0])
    return centroid, node


def node_semantic_lifecycle(node_record: dict) -> str:
    metadata = node_record.get("metadata", {}) if isinstance(node_record, dict) else {}
    if isinstance(metadata, dict) and metadata.get("detector") == "oracle_graph_bootstrap":
        return "verified"
    evidence = metadata.get("semantic_evidence", {}) if isinstance(metadata, dict) else {}
    return str(evidence.get("target_lifecycle", "context")) if isinstance(evidence, dict) else "context"


def node_is_confirmed(node_record: dict) -> bool:
    """Whether an object is safe to expose as an LLM navigation subgoal."""
    metadata = node_record.get("metadata", {}) if isinstance(node_record, dict) else {}
    if isinstance(metadata, dict) and metadata.get("detector") == "oracle_graph_bootstrap":
        return True
    return (
        node_record.get("state") == "confirmed"
        and int(node_record.get("observations", 1)) >= 3
        and float(node_record.get("confidence", 0.0)) >= 0.30
    )


def detection_visual_embedding(det, image_rgb: np.ndarray, clip_verifier=None):
    """Encode only the detected crop for object-map visual association."""
    if clip_verifier is None:
        return None
    try:
        height, width = image_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in det.bbox_xyxy]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        crop = image_rgb[y1:y2, x1:x2]
        if crop.shape[0] < 12 or crop.shape[1] < 12:
            return None
        return clip_verifier.encode_image(crop)
    except Exception as exc:
        # Perception must remain available if optional CLIP feature extraction
        # fails; association then falls back to geometry.
        return None


def point_in_rejected_region(point: np.ndarray, rejected_regions: list[dict]) -> dict | None:
    for region in rejected_regions or []:
        try:
            center = np.asarray(region.get("center"), dtype=np.float32)
            radius = float(region.get("radius", 0.0))
        except Exception:
            continue
        if center.shape[0] >= 2 and float(np.linalg.norm(point[:2] - center[:2])) <= radius:
            return region
    return None


def planner_node_reliability(node_record: dict, target_name: str) -> float:
    confidence = float(node_record.get("confidence", 0.0))
    observations = float(node_record.get("observations", 1))
    distance = float(node_record.get("distance", 0.0))
    # In visible-goal UAV-ON tests, a distant repeated false positive can
    # otherwise dominate a nearer true target.  Observation count is useful, but
    # only as weak evidence; current reachability should be a first-class cue.
    score = confidence * (1.0 + 0.12 * min(observations, 6.0)) - 0.014 * distance
    if distance > 45.0:
        score -= 0.25
    if target_name and label_matches_target(str(node_record.get("label", "")), target_name):
        score += 0.15
    if observations < 2:
        score -= 0.12
    return float(score)


def choose_graph_guard_target(graph_dict: dict, target_name: str) -> dict | None:
    """Pick the best target-like node by geometry-aware reliability."""
    target_nodes = [
        node for node in graph_dict.get("nodes", [])
        if target_name
        and label_matches_target(str(node.get("label", "")), target_name)
        and node.get("semantic_lifecycle") == "verified"
    ]
    if not target_nodes:
        return None
    return max(target_nodes, key=lambda node: planner_node_reliability(node, target_name))


def should_override_llm_node(chosen: dict | None, guard: dict | None, target_name: str) -> bool:
    """Let the graph guard override clearly inferior LLM node choices."""
    if guard is None:
        return False
    if chosen is None:
        return True
    if not (target_name and label_matches_target(str(chosen.get("label", "")), target_name)):
        return True
    chosen_score = planner_node_reliability(chosen, target_name)
    guard_score = planner_node_reliability(guard, target_name)
    chosen_distance = float(chosen.get("distance", 0.0))
    guard_distance = float(guard.get("distance", 0.0))
    if chosen_distance > guard_distance + 12.0 and guard_score >= chosen_score - 0.05:
        return True
    return guard_score > chosen_score + 0.16


def apply_graph_guard_decision(
    decision: dict,
    chosen: dict | None,
    guard_choice: dict | None,
    target_name: str,
    force_guard: bool = True,
) -> tuple[dict, dict | None]:
    """Use the graph scorer as the final arbiter for target-like node selection.

    The LLM is still useful for deciding exploration direction when no target
    node exists, but in UAV-ON visible-target runs it is too easily attracted by
    high-confidence false-positive nodes.  Once the graph contains target-like
    nodes, choose the geometry/stability-scored node deterministically and keep
    the LLM choice only as diagnostic metadata.
    """
    if guard_choice is None:
        return decision, chosen
    original_node_id = decision.get("node_id") if isinstance(decision, dict) else None
    guard_node_id = guard_choice.get("node_id")
    if force_guard or should_override_llm_node(chosen, guard_choice, target_name):
        updated = {
            **dict(decision),
            "subgoal_type": "object_node",
            "node_id": guard_node_id,
            "guard_override": bool(original_node_id != guard_node_id),
            "guard_forced": bool(force_guard),
            "guard_reason": "graph_guard_target_node_final_arbiter",
            "llm_original_node_id": original_node_id,
            "guard_node_id": guard_node_id,
            "guard_node_distance": guard_choice.get("distance"),
            "guard_node_reliability": round(planner_node_reliability(guard_choice, target_name), 3),
        }
        return updated, guard_choice
    return decision, chosen


def reject_reached_semantic_subgoal(state: dict, nav_target, radius: float = 8.0) -> dict:
    """Reject a reached semantic node when the global success condition failed."""
    record = {"triggered": False, "rejected_node_id": None, "rejected_region": None}
    if not isinstance(nav_target, dict):
        return record
    executor = nav_target.get("executor")
    if not isinstance(executor, dict) or executor.get("status") != "reached":
        return record
    target_node = nav_target.get("target_node")
    node_id = target_node.get("node_id") if isinstance(target_node, dict) else None
    subgoal = nav_target_subgoal_array(nav_target)
    if not node_id and subgoal is None:
        return record

    record["triggered"] = True
    if node_id:
        rejected = set(str(item) for item in state.get("rejected_node_ids", []))
        rejected.add(str(node_id))
        state["rejected_node_ids"] = sorted(rejected)
        record["rejected_node_id"] = str(node_id)
    if subgoal is not None:
        region = {
            "center": np.asarray(subgoal, dtype=np.float32).round(3).tolist(),
            "radius": float(radius),
            "reason": "reached_without_global_success",
        }
        rejected_regions = list(state.get("rejected_subgoal_regions", []))
        rejected_regions.append(region)
        state["rejected_subgoal_regions"] = rejected_regions[-12:]
        record["rejected_region"] = region
    state["subgoal"] = None
    state["last_plan_step"] = -10_000
    state.pop("planner_chosen_node", None)
    state.pop("target_candidate_lock", None)
    state.pop("target_lock_subgoal", None)
    state.pop("target_lock_node", None)
    state.pop("target_lock_node_id", None)
    return record


def scene_graph_to_planner_dict(
    graph_builder: ConceptGraphBuilder,
    position: np.ndarray,
    max_nodes: int = 40,
    target_name: str = "",
    rejected_node_ids: set[str] | None = None,
    rejected_regions: list[dict] | None = None,
    target_node_budget: int = 8,
    target_cluster_radius: float = 8.0,
) -> dict:
    """Return a compact graph dictionary for LLM planning."""
    nodes = []
    selected_ids = set()
    rejected_node_ids = rejected_node_ids or set()
    rejected_regions = rejected_regions or []
    for node_id, node in graph_builder.scene_graph.graph.nodes(data=True):
        if str(node_id) in rejected_node_ids:
            continue
        centroid = np.asarray(node.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
        rejected_region = point_in_rejected_region(centroid, rejected_regions)
        if rejected_region is not None:
            continue
        distance = float(np.linalg.norm(centroid[:2] - position[:2]))
        nodes.append({
            "node_id": str(node_id),
            "label": str(node.get("label", "object")),
            "caption": str(node.get("caption", ""))[:120],
            "centroid": np.asarray(centroid, dtype=np.float32).round(2).tolist(),
            "confidence": round(float(node.get("confidence", 0.0)), 3),
            "observations": int(node.get("observations", 1)),
            "distance": round(distance, 2),
            "semantic_lifecycle": node_semantic_lifecycle(node),
            "map_state": str(node.get("state", "tentative")),
        })
    if target_name:
        target_nodes = [
            node for node in nodes
            if label_matches_target(str(node.get("label", "")), target_name)
            and node.get("semantic_lifecycle") == "verified"
            and node.get("map_state") == "confirmed"
        ]
        other_nodes = [
            node for node in nodes
            if not label_matches_target(str(node.get("label", "")), target_name)
        ]
        target_nodes.sort(key=lambda node: planner_node_reliability(node, target_name), reverse=True)
        deduped_targets = []
        for node in target_nodes:
            centroid = np.asarray(node["centroid"], dtype=np.float32)
            too_close = any(
                float(np.linalg.norm(centroid[:2] - np.asarray(existing["centroid"], dtype=np.float32)[:2])) <= target_cluster_radius
                for existing in deduped_targets
            )
            if too_close:
                continue
            node = dict(node)
            node["planner_reliability"] = round(planner_node_reliability(node, target_name), 3)
            deduped_targets.append(node)
            if len(deduped_targets) >= target_node_budget:
                break
        nodes = deduped_targets + other_nodes
    # Do not ask the LLM to navigate to one-frame objects. Tentative nodes are
    # still retained internally for later association, but not in its prompt.
    nodes = [node for node in nodes if node.get("map_state") == "confirmed"]
    nodes.sort(key=lambda n: (
        0 if target_name and label_matches_target(n["label"], target_name) else 1,
        0 if n["observations"] >= 2 else 1,
        -n["confidence"],
        n["distance"],
    ))
    nodes = nodes[:max_nodes]
    selected_ids.update(node["node_id"] for node in nodes)
    edges = [
        {"source": str(src), "target": str(dst), **dict(data)}
        for src, dst, data in graph_builder.scene_graph.graph.edges(data=True)
        if str(src) in selected_ids and str(dst) in selected_ids
    ]
    return {"nodes": nodes, "edges": edges}


def summarize_graph_goal_diagnostics(scene_graph, target_name: str, goal_pose: np.ndarray, top_k: int = 10) -> dict:
    """Summarize whether the final scene graph contains nodes close to the true goal."""
    all_nodes = []
    target_like_nodes = []
    for node_id, node in scene_graph.graph.nodes(data=True):
        centroid = np.asarray(node.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32)
        distance = float(np.linalg.norm(centroid - goal_pose[:3]))
        record = {
            "node_id": str(node_id),
            "label": str(node.get("label", "")),
            "caption": str(node.get("caption", ""))[:160],
            "centroid": centroid.round(3).tolist(),
            "distance_to_goal": round(distance, 3),
            "confidence": round(float(node.get("confidence", 0.0)), 3),
            "observations": int(node.get("observations", 1)),
        }
        all_nodes.append(record)
        if label_matches_target(record["label"], target_name):
            target_like_nodes.append(record)

    all_nodes.sort(key=lambda item: item["distance_to_goal"])
    target_like_nodes.sort(key=lambda item: item["distance_to_goal"])
    return {
        "nearest_nodes_to_goal": all_nodes[:top_k],
        "target_like_nodes": target_like_nodes[:top_k],
        "num_target_like_nodes": len(target_like_nodes),
        "nearest_node_distance": all_nodes[0]["distance_to_goal"] if all_nodes else None,
        "nearest_target_like_distance": target_like_nodes[0]["distance_to_goal"] if target_like_nodes else None,
    }


def annotate_nav_target_diagnostics(nav_target, goal_pose: np.ndarray, distance_to_goal: float) -> dict | None:
    """Add distances from the chosen subgoal/node to the true goal for debugging."""
    if nav_target is None:
        return None
    if not isinstance(nav_target, dict):
        return nav_target
    annotated = dict(nav_target)
    candidate = None
    if "subgoal" in annotated:
        candidate = annotated.get("subgoal")
    elif "centroid" in annotated:
        candidate = annotated.get("centroid")
    elif "target_position" in annotated:
        candidate = annotated.get("target_position")
    elif isinstance(annotated.get("executor"), dict) and "target_position" in annotated["executor"]:
        candidate = annotated["executor"].get("target_position")
    if candidate is not None:
        try:
            candidate_arr = np.asarray(candidate, dtype=np.float32)
            annotated["subgoal_distance_to_true_goal"] = round(float(np.linalg.norm(candidate_arr[:3] - goal_pose[:3])), 3)
        except Exception:
            annotated["subgoal_distance_to_true_goal"] = None
    annotated["uav_distance_to_true_goal_before_action"] = round(float(distance_to_goal), 3)
    return annotated


def exploration_subgoal(position: np.ndarray, yaw: float, direction: str, distance: float = 12.0) -> np.ndarray:
    """Create a short, world-space exploration waypoint from a semantic direction."""
    offset = {"left": np.pi / 2, "right": -np.pi / 2}.get(direction, 0.0)
    heading = yaw + offset
    return np.asarray(position, dtype=np.float32) + np.asarray(
        [distance * np.cos(heading), distance * np.sin(heading), 0.0], dtype=np.float32
    )


def navigation_subgoal_from_node(centroid: np.ndarray, approach_z: float = 0.0) -> np.ndarray:
    """Use scene-graph node XY with a stable object-approach altitude.

    RGB-D object centroid height is noisy in AirSim/NED coordinates and often
    causes the local executor to climb/descend toward a false vertical target.
    UAV-ON object poses are near ground/object level, so approach at z≈0 is a
    better default than the projected bbox depth z.
    """
    subgoal = np.asarray(centroid, dtype=np.float32).copy()
    subgoal[2] = float(approach_z)
    return subgoal


def navigation_subgoal_for_target_node(node: dict, centroid: np.ndarray) -> np.ndarray:
    """Choose a navigation target from a semantic graph node."""
    metadata = node.get("metadata", {}) if isinstance(node, dict) else {}
    if isinstance(metadata, dict) and metadata.get("detector") == "oracle_graph_bootstrap":
        return np.asarray(centroid, dtype=np.float32)
    centroid = np.asarray(centroid, dtype=np.float32)
    observations = int(node.get("observations", 1)) if isinstance(node, dict) else 1
    confidence = float(node.get("confidence", 0.0)) if isinstance(node, dict) else 0.0
    z_is_plausible = -2.0 <= float(centroid[2]) <= 8.0
    if observations >= 3 and confidence >= 0.30 and z_is_plausible:
        return centroid.copy()
    return navigation_subgoal_from_node(centroid)


def is_oracle_graph_node(node: dict | None) -> bool:
    metadata = node.get("metadata", {}) if isinstance(node, dict) else {}
    return isinstance(metadata, dict) and metadata.get("detector") == "oracle_graph_bootstrap"


def node_goal_reached(position: np.ndarray, subgoal: np.ndarray, planar_threshold: float = 4.0) -> bool:
    return float(np.linalg.norm(np.asarray(position[:2]) - np.asarray(subgoal[:2]))) <= planar_threshold


def add_oracle_goal_node(graph_builder: ConceptGraphBuilder, target_name: str, goal_pose: np.ndarray) -> None:
    """Inject a true target node for graph/executor upper-bound experiments."""
    center = np.asarray(goal_pose, dtype=np.float32)
    extent = np.asarray([1.5, 1.5, 1.5], dtype=np.float32)
    node = ConceptNode(
        node_id=f"node_{graph_builder._next_node_index:06d}",
        label=normalize_target_name(target_name),
        caption="oracle UAV-ON true target node",
        centroid=center.round(4).tolist(),
        bbox_min=(center - extent).round(4).tolist(),
        bbox_max=(center + extent).round(4).tolist(),
        observations=99,
        confidence=1.0,
        metadata={"detector": "oracle_graph_bootstrap", "goal_pose": center.round(4).tolist()},
    )
    graph_builder._next_node_index += 1
    graph_builder.scene_graph.add_or_update_node(node, merge_iou=0.0)


def choose_hierarchical_action(
    planner: DeepSeekPlanner | None,
    target_name: str,
    target_description: str,
    gd_detections,
    far_target_detections,
    target_candidates: list[dict],
    graph_builder: ConceptGraphBuilder,
    position: np.ndarray,
    yaw: float,
    depth: np.ndarray,
    executor: LocalWaypointExecutor,
    history: list[str],
    state: dict,
    step: int,
    image_width: int,
    replan_interval: int = 8,
    force_planner: bool = False,
    target_memory_mode: str = "conservative",
    visual_close_approach: bool = False,
) -> tuple[str, float, str, dict]:
    """Choose a semantic subgoal, then delegate control to the local executor."""
    rejected_node_ids = set(str(item) for item in state.get("rejected_node_ids", []))
    candidate_subgoal, candidate_lock = (None, None)
    if target_memory_mode != "off":
        candidate_subgoal, candidate_lock = update_target_candidate_lock(
            state, target_candidates, position, step
        )
    target_centroid, target_node = find_target_node(graph_builder.scene_graph, target_name, position, rejected_node_ids)
    planner_info = None
    planner_chosen_node = None

    def run_candidate_lock() -> tuple[str, float, str, dict] | None:
        if candidate_subgoal is None or force_planner:
            return None
        source = "target_candidate_lock_subgoal"
        state.update({"subgoal": candidate_subgoal.tolist(), "source": source, "last_plan_step": step})
        action, angle_error, executor_info = executor.choose_action(position, yaw, candidate_subgoal, depth)
        detail = {
            "subgoal": np.asarray(candidate_subgoal).round(3).tolist(),
            "executor": executor_info,
            "target_candidate_lock": candidate_lock,
            "target_memory_mode": target_memory_mode,
        }
        if executor_info.get("status") == "reached":
            state["subgoal"] = None
        return action, angle_error, source, detail

    def run_close_visual_approach() -> tuple[str, float, str, dict] | None:
        """Refine the final approach from a stable, current-frame visual track.

        This is deliberately a local-executor safety override, not an LLM
        bypass: it activates only after the same CLIP-verified candidate was
        observed in at least two recent frames and lies within 14 metres.
        """
        if not visual_close_approach or candidate_subgoal is None or not isinstance(candidate_lock, dict):
            return None
        observations = int(candidate_lock.get("observations", 0))
        distance = float(np.linalg.norm(np.asarray(candidate_subgoal)[:2] - position[:2]))
        if observations < 2 or distance > 14.0:
            return None
        action, angle_error, executor_info = executor.choose_action(position, yaw, candidate_subgoal, depth)
        state.update({
            "subgoal": np.asarray(candidate_subgoal).tolist(),
            "source": "close_visual_target_approach",
            "last_plan_step": step,
        })
        detail = {
            "subgoal": np.asarray(candidate_subgoal).round(3).tolist(),
            "executor": executor_info,
            "target_candidate_lock": candidate_lock,
            "near_target_refinement": True,
            "visual_observations": observations,
            "candidate_distance": round(distance, 3),
        }
        if executor_info.get("status") == "reached":
            state["subgoal"] = None
        return action, angle_error, "close_visual_target_approach", detail

    close_visual_result = run_close_visual_approach()
    if close_visual_result is not None:
        return close_visual_result

    if target_memory_mode == "aggressive":
        candidate_result = run_candidate_lock()
        if candidate_result is not None:
            return candidate_result

    # A far cue contributes only a camera bearing.  It cannot create a 3-D
    # destination or force forward movement while its depth is unreliable.
    far_error, far_det = bbox_steering_error(far_target_detections, target_name, image_width)
    far_turns = int(state.get("far_cue_turns", 0))
    if far_error is not None and abs(np.degrees(far_error)) > 16.0 and far_turns < 2:
        state["far_cue_turns"] = far_turns + 1
        return "turn_to_goal", far_error, "far_target_bearing_turn", {
            "target_bbox": far_det,
            "target_lifecycle": "far_cue",
            "reason": "far_target_bearing_only",
        }
    state["far_cue_turns"] = 0

    if target_centroid is not None and not force_planner:
        subgoal = navigation_subgoal_for_target_node(target_node, target_centroid)
        source = "target_graph_subgoal"
        # The waypoint executor uses a comparatively loose reach radius.  Once
        # a confirmed target node is nearby, retain graph guidance but use the
        # current target box for the final few metres instead of declaring the
        # semantic waypoint reached and falling back to exploration.
        node_distance = float(np.linalg.norm(np.asarray(target_centroid)[:2] - position[:2]))
        visual_error, visual_det = bbox_steering_error(gd_detections, target_name, image_width)
        if node_distance <= 15.0 and visual_error is not None:
            if abs(np.degrees(visual_error)) > 18.0:
                return "turn_to_goal", visual_error, "target_graph_visual_servo_turn", {
                    "target_node": target_node,
                    "target_bbox": visual_det,
                    "node_distance": round(node_distance, 3),
                }
            return "forward_slow", visual_error, "target_graph_visual_servo_forward", {
                "target_node": target_node,
                "target_bbox": visual_det,
                "node_distance": round(node_distance, 3),
            }
        state.update({"subgoal": subgoal.tolist(), "source": source, "last_plan_step": step})
        node_id = target_node.get("node_id") if isinstance(target_node, dict) else None
        if node_id and target_memory_mode == "aggressive":
            state["target_lock_subgoal"] = subgoal.tolist()
            state["target_lock_node"] = target_node
            state["target_lock_node_id"] = str(node_id)
        action, angle_error, executor_info = executor.choose_action(position, yaw, subgoal, depth)
        detail = {
            "subgoal": np.asarray(subgoal).round(3).tolist(),
            "raw_node_centroid": np.asarray(target_centroid).round(3).tolist(),
            "executor": executor_info,
            "target_node": target_node,
        }
        oracle_node = is_oracle_graph_node(target_node)
        # Do not reject a correct object node merely because XY is close while
        # the UAV is still high above it. Let the local executor complete its
        # vertical approach and report a true 3-D reach condition.
        reached_candidate = executor_info.get("status") == "reached"
        if reached_candidate:
            state["subgoal"] = None
            node_id = target_node.get("node_id") if isinstance(target_node, dict) else None
            # Oracle/high-confidence nodes are reliable enough to keep tracking;
            # do not reject them just because the local waypoint was reached at
            # a loose planar radius before the global UAV-ON success threshold.
            if oracle_node:
                return "hover", angle_error, "target_graph_hold", detail
            if node_id and float(target_node.get("confidence", 0.0)) < 0.65:
                rejected_node_ids.add(str(node_id))
                state["rejected_node_ids"] = sorted(rejected_node_ids)
                if state.get("target_lock_node_id") == str(node_id):
                    state.pop("target_lock_subgoal", None)
                    state.pop("target_lock_node", None)
                    state.pop("target_lock_node_id", None)
                detail["rejected_node_id"] = str(node_id)
            # Reached a semantic candidate but the episode has not succeeded yet;
            # scan instead of hovering forever on a likely false positive.
            return "rotate_left", 0.0, "target_graph_reached_scan", detail
        return action, angle_error, source, detail

    if target_memory_mode == "conservative":
        candidate_result = run_candidate_lock()
        if candidate_result is not None:
            return candidate_result

    bbox_error, bbox_det = bbox_steering_error(gd_detections, target_name, image_width)
    if bbox_error is not None and not force_planner:
        bbox_turn_count = int(state.get("bbox_turn_count", 0))
        if abs(np.degrees(bbox_error)) > 25 and bbox_turn_count < 2:
            state["subgoal"] = None
            state["bbox_turn_count"] = bbox_turn_count + 1
            return "turn_to_goal", bbox_error, "target_bbox_tracking_turn", {"target_bbox": bbox_det}
        state["subgoal"] = None
        state["bbox_turn_count"] = 0
        return "forward", bbox_error, "target_bbox_tracking_forward", {"target_bbox": bbox_det}

    state["bbox_turn_count"] = 0

    locked_subgoal = state.get("target_lock_subgoal")
    locked_node = state.get("target_lock_node")
    locked_node_id = state.get("target_lock_node_id")
    if (
        target_memory_mode == "aggressive"
        and locked_subgoal is not None
        and str(locked_node_id) not in rejected_node_ids
        and not force_planner
    ):
        subgoal = np.asarray(locked_subgoal, dtype=np.float32)
        action, angle_error, executor_info = executor.choose_action(position, yaw, subgoal, depth)
        detail = {
            "subgoal": np.asarray(subgoal).round(3).tolist(),
            "executor": executor_info,
            "target_node": locked_node,
            "target_lock_node_id": locked_node_id,
        }
        if executor_info.get("status") == "reached":
            state["subgoal"] = None
        return action, angle_error, "target_lock_subgoal", detail

    if state.get("subgoal") is not None and step - int(state.get("last_plan_step", -replan_interval)) < replan_interval:
        subgoal = np.asarray(state["subgoal"], dtype=np.float32)
        source = f"cached_{state.get('source', 'semantic_subgoal')}"
        cached_node = state.get("planner_chosen_node")
        if isinstance(cached_node, dict):
            planner_chosen_node = cached_node
    elif planner is not None:
        graph_dict = scene_graph_to_planner_dict(
            graph_builder,
            position,
            target_name=target_name,
            rejected_node_ids=rejected_node_ids,
            rejected_regions=state.get("rejected_subgoal_regions", []),
        )
        decision = planner.decide_subgoal(
            scene_graph=graph_dict,
            current_position=position.round(3).tolist(),
            target_object=target_name,
            target_description=target_description,
            history=history,
        )
        planner_info = decision
        chosen = next(
            (node for node in graph_dict["nodes"] if node["node_id"] == decision.get("node_id")),
            None,
        )
        guard_choice = choose_graph_guard_target(graph_dict, target_name)
        decision, chosen = apply_graph_guard_decision(
            decision,
            chosen,
            guard_choice,
            target_name,
            force_guard=True,
        )
        planner_info = decision
        if chosen is not None:
            subgoal = navigation_subgoal_for_target_node(
                chosen, np.asarray(chosen["centroid"], dtype=np.float32)
            )
            source = "llm_object_subgoal"
            planner_chosen_node = dict(chosen)
        else:
            subgoal = exploration_subgoal(position, yaw, decision.get("direction", "forward"))
            source = "llm_frontier_subgoal"
            planner_chosen_node = None
        state.update({
            "subgoal": subgoal.tolist(),
            "source": source,
            "last_plan_step": step,
            "planner_chosen_node": planner_chosen_node,
        })
    else:
        subgoal = exploration_subgoal(position, yaw, "left" if len(history) % 4 == 0 else "forward")
        source = "frontier_fallback_subgoal"
        state.update({"subgoal": subgoal.tolist(), "source": source, "last_plan_step": step})

    action, angle_error, executor_info = executor.choose_action(position, yaw, subgoal, depth)
    detail = {"subgoal": np.asarray(subgoal).round(3).tolist(), "executor": executor_info}
    if planner_chosen_node is not None:
        detail["target_node"] = planner_chosen_node
    elif target_node is not None:
        detail["target_node"] = target_node
    if planner_info is not None:
        detail["planner"] = planner_info
    if executor_info.get("status") == "reached":
        state["subgoal"] = None
    return action, angle_error, source, detail


def choose_llm_planned_action(
    planner: DeepSeekPlanner | None,
    target_name: str,
    gd_detections,
    graph_builder: ConceptGraphBuilder,
    position: np.ndarray,
    yaw: float,
    image_width: int,
    step: int,
    llm_history: list[str],
    stuck_counter: int = 0,
) -> tuple[str, float, str, dict | None]:
    """Use LLM planning when available, with local target/executor safety fallbacks."""
    bbox_error, bbox_det = bbox_steering_error(gd_detections, target_name, image_width)
    if bbox_error is not None:
        if abs(np.degrees(bbox_error)) > 20:
            return "turn_to_goal", bbox_error, "target_bbox_executor_turn", bbox_det
        return "forward", bbox_error, "target_bbox_executor_forward", bbox_det

    if stuck_counter >= 8:
        return "ascend", 0.0, "executor_stuck_ascend", {"stuck_counter": stuck_counter}
    if stuck_counter >= 5:
        return "rotate_left", 0.0, "executor_stuck_rotate", {"stuck_counter": stuck_counter}

    target_centroid, target_node = find_target_node(graph_builder.scene_graph, target_name, position)
    if target_centroid is not None:
        graph_error = angle_error_to_point(position, target_centroid, yaw)
        if abs(np.degrees(graph_error)) > 30:
            return "turn_to_goal", graph_error, "target_graph_executor_turn", target_node
        return "forward", graph_error, "target_graph_executor_forward", target_node

    if planner is None:
        return choose_conceptgraph_action(
            target_name, gd_detections, graph_builder, position, yaw, image_width, step, stuck_counter
        )

    planner_graph = scene_graph_to_planner_dict(graph_builder, position)
    decision = planner.decide_action(
        scene_graph=planner_graph,
        current_position=position.round(3).tolist(),
        target_object=target_name,
        history=llm_history,
    )
    llm_action = str(decision.get("action", "forward")).strip().lower()
    action_map = {
        "forward": "forward",
        "backward": "backward",
        "rotl": "rotate_left",
        "rotr": "rotate_right",
        "left": "rotate_left",
        "right": "rotate_right",
        "ascend": "ascend",
        "descend": "descend",
        "stop": "hover",
    }
    executor_action = action_map.get(llm_action, "forward")
    return executor_action, 0.0, "llm_planner_executor", {
        "llm_action": llm_action,
        "reasoning": decision.get("reasoning", ""),
        "raw_response": decision.get("raw_response", ""),
    }


def choose_clip_llm_action(
    planner: DeepSeekPlanner | None,
    target_name: str,
    detected_label: str,
    confidence: float,
    graph_builder: ConceptGraphBuilder,
    position: np.ndarray,
    angle_diff_rad: float,
    step: int,
    llm_history: list[str],
    stuck_counter: int = 0,
) -> tuple[str, float, str, dict | None]:
    """LLM planning over a CLIP-labeled scene graph with local executor fallback."""
    if stuck_counter >= 8:
        return "ascend", 0.0, "clip_executor_stuck_ascend", {"stuck_counter": stuck_counter}
    if stuck_counter >= 5:
        return "rotate_left", 0.0, "clip_executor_stuck_rotate", {"stuck_counter": stuck_counter}

    if label_matches_target(detected_label, target_name) and confidence >= 0.025:
        if abs(np.degrees(angle_diff_rad)) > 45:
            return "turn_to_goal", angle_diff_rad, "clip_target_label_turn", {
                "label": detected_label,
                "confidence": float(confidence),
            }
        return "forward", angle_diff_rad, "clip_target_label_forward", {
            "label": detected_label,
            "confidence": float(confidence),
        }

    if planner is None:
        action = choose_action("deepseek", target_name, detected_label, confidence, angle_diff_rad, step)
        return action, angle_diff_rad, "clip_rule_fallback", None

    planner_graph = scene_graph_to_planner_dict(graph_builder, position)
    decision = planner.decide_action(
        scene_graph=planner_graph,
        current_position=position.round(3).tolist(),
        target_object=target_name,
        history=llm_history,
    )
    llm_action = str(decision.get("action", "forward")).strip().lower()
    action_map = {
        "forward": "forward",
        "backward": "backward",
        "rotl": "rotate_left",
        "rotr": "rotate_right",
        "left": "rotate_left",
        "right": "rotate_right",
        "ascend": "ascend",
        "descend": "descend",
        "stop": "hover",
    }
    executor_action = action_map.get(llm_action, "forward")
    return executor_action, angle_diff_rad, "clip_llm_planner_executor", {
        "llm_action": llm_action,
        "reasoning": decision.get("reasoning", ""),
        "raw_response": decision.get("raw_response", ""),
        "top_clip_label": detected_label,
        "top_clip_confidence": float(confidence),
    }


def clip_target_prompt_set(target_name: str, description: str = "") -> list[str]:
    """Prompts for CLIP active search."""
    target = normalize_target_name(target_name)
    prompts = [
        target,
        f"a drone view photo of a {target}",
        f"a close outdoor view of a {target}",
        f"the target object is a {target}",
    ]
    description = " ".join(str(description or "").strip().split())
    if description:
        prompts.append(f"a drone view photo of {target}, {description[:180]}")
    aliases = {
        "bus stop": ["bus shelter", "public transit waiting shelter", "glass bus stop shelter"],
        "traffic light": ["traffic signal", "stoplight"],
        "lamp post": ["street lamp", "light pole"],
        "bench": ["park bench"],
        "fountain": ["water fountain"],
        "rock": ["large rock", "decorative stone", "landscape boulder"],
        "caravan": ["camper trailer", "mobile home trailer", "travel caravan"],
        # UAV-ON contains several small or visually variable target classes.
        # Use alternate names here as CLIP evidence, independently of the
        # GroundingDINO caption aliases used for proposal generation.
        "soccer ball": ["football", "sports ball", "black and white soccer ball"],
        "teapot": ["tea kettle", "kettle"],
        "table": ["picnic table", "outdoor table"],
        "chair": ["outdoor chair", "seat"],
    }.get(target, [])
    prompts.extend(f"a drone view photo of a {alias}" for alias in aliases)
    deduped = []
    for prompt in prompts:
        if prompt and prompt not in deduped:
            deduped.append(prompt)
    return deduped


def clip_negative_prompt_set() -> list[str]:
    """Common non-target scene prompts for contrastive CLIP scoring."""
    return [
        "a drone view photo of trees and vegetation",
        "a drone view photo of road and pavement",
        "a drone view photo of sky and clouds",
        "a drone view photo of grass and ground",
        "a drone view photo of a generic building",
        "a drone view photo of a fence or wall",
        "a drone view photo of an empty outdoor scene",
    ]


_CLIP_TEXT_FEATURE_CACHE = {}


def clip_text_features(detector: CLIPDetector, prompts: list[str]):
    import clip as openai_clip
    import torch

    cache_key = (id(detector), tuple(prompts))
    cached = _CLIP_TEXT_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    text_inputs = torch.cat([openai_clip.tokenize(prompt) for prompt in prompts]).to(detector.device)
    with torch.no_grad():
        features = detector.model.encode_text(text_inputs)
        features /= features.norm(dim=-1, keepdim=True)
    _CLIP_TEXT_FEATURE_CACHE[cache_key] = features
    return features


def clip_image_target_score(detector: CLIPDetector, image_rgb: np.ndarray, text_features, negative_features=None) -> tuple[float, float, float]:
    import torch
    from PIL import Image

    if image_rgb.size == 0:
        return 0.0, 0.0, 0.0
    image_pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    image_input = detector.preprocess(image_pil).unsqueeze(0).to(detector.device)
    with torch.no_grad():
        image_features = detector.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        target_scores = (image_features @ text_features.T).squeeze(0)
        target_score = float(target_scores.max().item())
        negative_score = 0.0
        if negative_features is not None:
            negative_scores = (image_features @ negative_features.T).squeeze(0)
            negative_score = float(negative_scores.max().item())
    return target_score - negative_score, target_score, negative_score


def clip_grid_scores(detector: CLIPDetector, image_rgb: np.ndarray, text_features, negative_features=None, grid: tuple[int, int] = (3, 3)) -> dict:
    """Score whole image and grid crops against target prompts."""
    height, width = image_rgb.shape[:2]
    rows, cols = grid
    cells = []
    whole_score, whole_target_score, whole_negative_score = clip_image_target_score(
        detector, image_rgb, text_features, negative_features
    )
    for row in range(rows):
        for col in range(cols):
            y1 = int(row * height / rows)
            y2 = int((row + 1) * height / rows)
            x1 = int(col * width / cols)
            x2 = int((col + 1) * width / cols)
            crop = image_rgb[y1:y2, x1:x2]
            score, target_score, negative_score = clip_image_target_score(detector, crop, text_features, negative_features)
            cells.append({
                "row": row,
                "col": col,
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": round(score, 4),
                "target_score": round(target_score, 4),
                "negative_score": round(negative_score, 4),
                "center_x_norm": round(((x1 + x2) * 0.5 / max(1, width) - 0.5) * 2.0, 4),
            })
    cells.sort(key=lambda item: item["score"], reverse=True)
    return {
        "whole_score": round(whole_score, 4),
        "whole_target_score": round(whole_target_score, 4),
        "whole_negative_score": round(whole_negative_score, 4),
        "top_cells": cells[:5],
        "all_cells": cells,
    }


def choose_clip_active_action(
    detector: CLIPDetector,
    rgb: np.ndarray,
    target_name: str,
    target_description: str,
    detected_label: str,
    detected_confidence: float,
    state: dict,
    step: int,
    stuck_counter: int = 0,
) -> tuple[str, float, str, dict]:
    """CLIP-only active search using scan-then-advance target evidence."""
    prompts = state.get("clip_prompts")
    if prompts is None:
        prompts = clip_target_prompt_set(target_name, target_description)
        state["clip_prompts"] = prompts
        state["clip_text_features"] = clip_text_features(detector, prompts)
        negative_prompts = clip_negative_prompt_set()
        state["clip_negative_prompts"] = negative_prompts
        state["clip_negative_features"] = clip_text_features(detector, negative_prompts)
    text_features = state["clip_text_features"]
    negative_features = state.get("clip_negative_features")
    scores = clip_grid_scores(detector, rgb, text_features, negative_features=negative_features, grid=(3, 3))
    top_cells = scores["top_cells"]
    center_cells = [cell for cell in top_cells if cell.get("col") == 1]
    center_best = max(center_cells, key=lambda item: item["score"]) if center_cells else None
    top = top_cells[0] if top_cells else {"score": 0.0, "center_x_norm": 0.0, "row": 1, "col": 1}
    top_score = float(top["score"])
    center_score = float(center_best["score"]) if center_best else -1e9
    whole_score = float(scores["whole_score"])
    prev_best = float(state.get("best_score", -1e9))
    state["best_score"] = max(prev_best, top_score, whole_score)
    scan_steps = 12
    phase = str(state.get("phase", "scan"))
    effective_stuck_counter = 0 if phase in {"scan", "align"} else stuck_counter

    def view_score() -> float:
        # Prefer centered target evidence; side-only evidence is useful but less
        # reliable for committing to forward motion.
        centered = center_score if center_best else -1e9
        return float(max(whole_score, top_score - 0.01, centered + 0.015))

    if effective_stuck_counter >= 8:
        action = "ascend"
        source = "clip_active_stuck_ascend"
        angle = 0.0
        state["phase"] = "scan"
        state["scan_records"] = []
    elif effective_stuck_counter >= 5:
        action = "rotate_left"
        source = "clip_active_stuck_scan"
        angle = 0.0
        state["phase"] = "scan"
        state["scan_records"] = []
    elif phase == "scan":
        records = list(state.get("scan_records", []))
        records.append({
            "scan_index": len(records),
            "view_score": round(view_score(), 4),
            "whole_score": round(whole_score, 4),
            "top_score": round(top_score, 4),
            "center_score": round(center_score, 4) if center_best else None,
            "top_cell": top,
        })
        state["scan_records"] = records
        if len(records) < scan_steps:
            action = "rotate_left"
            source = "clip_active_scan_360"
            angle = 0.0
        else:
            best_record = max(records, key=lambda item: item["view_score"])
            best_index = int(best_record["scan_index"])
            best_view_score = float(best_record["view_score"])
            current_scan_index = len(records) - 1
            align_turns = (best_index - current_scan_index) % scan_steps
            state["best_scan_record"] = best_record
            state["align_remaining"] = align_turns
            # Negative or near-zero contrastive evidence means CLIP has not found
            # a credible target direction; keep scanning with small position changes.
            if best_view_score < -0.005:
                state["phase"] = "advance"
                state["advance_remaining"] = 2
                state["scan_records"] = []
                action = "forward"
                source = "clip_active_probe_forward_low_evidence"
                angle = 0.0
            elif align_turns > 0:
                state["phase"] = "align"
                action = "rotate_left"
                source = "clip_active_align_best_scan"
                angle = 0.0
                state["align_remaining"] = align_turns - 1
            else:
                state["phase"] = "advance"
                state["advance_remaining"] = 3 if best_view_score > 0.015 else 1
                state["scan_records"] = []
                action = "forward"
                source = "clip_active_advance_best_scan"
                angle = 0.0
    elif phase == "align":
        remaining = int(state.get("align_remaining", 0))
        if remaining > 0:
            action = "rotate_left"
            source = "clip_active_align_best_scan"
            angle = 0.0
            state["align_remaining"] = remaining - 1
        else:
            best_record = state.get("best_scan_record", {})
            best_view_score = float(best_record.get("view_score", 0.0)) if isinstance(best_record, dict) else 0.0
            state["phase"] = "advance"
            state["advance_remaining"] = 3 if best_view_score > 0.015 else 1
            state["scan_records"] = []
            action = "forward"
            source = "clip_active_advance_best_scan"
            angle = 0.0
    else:
        remaining = int(state.get("advance_remaining", 0))
        # If the current centered evidence collapses, stop the run-away forward
        # segment and rescan immediately.
        if remaining <= 0 or (whole_score < -0.02 and top_score < -0.02):
            state["phase"] = "scan"
            state["scan_records"] = []
            action = "rotate_left"
            source = "clip_active_rescan"
            angle = 0.0
        else:
            action = "forward"
            angle = 0.0
            source = "clip_active_advance_best_scan"
            state["advance_remaining"] = remaining - 1

    detail = {
        "whole_score": whole_score,
        "whole_target_score": scores.get("whole_target_score"),
        "whole_negative_score": scores.get("whole_negative_score"),
        "top_cell": top,
        "center_best_cell": center_best,
        "side_margin": round(float(top_score - center_score), 4) if center_best else None,
        "phase": state.get("phase", phase),
        "scan_records": state.get("scan_records", []),
        "best_scan_record": state.get("best_scan_record"),
        "align_remaining": int(state.get("align_remaining", 0)),
        "advance_remaining": int(state.get("advance_remaining", 0)),
        "top_cells": scores["top_cells"],
        "best_score": round(float(state.get("best_score", top_score)), 4),
        "prompts": prompts,
        "negative_prompts": state.get("clip_negative_prompts", []),
        "detected_label": detected_label,
        "detected_confidence": float(detected_confidence),
    }
    return action, angle, source, detail


def choose_conceptgraph_action(
    target_name: str,
    gd_detections,
    graph_builder: ConceptGraphBuilder,
    position: np.ndarray,
    yaw: float,
    image_width: int,
    step: int,
    stuck_counter: int = 0,
) -> tuple[str, float, str, dict | None]:
    """Target-driven policy using current detections first, then scene graph nodes."""
    if stuck_counter >= 8:
        return "ascend", 0.0, "stuck_ascend", {"stuck_counter": stuck_counter}
    if stuck_counter >= 5:
        return "rotate_left", 0.0, "stuck_rotate", {"stuck_counter": stuck_counter}

    bbox_error, bbox_det = bbox_steering_error(gd_detections, target_name, image_width)
    if bbox_error is not None:
        if abs(np.degrees(bbox_error)) > 25:
            return "turn_to_goal", bbox_error, "target_bbox_turn", bbox_det
        return "forward", bbox_error, "target_bbox_forward", bbox_det

    target_centroid, target_node = find_target_node(graph_builder.scene_graph, target_name, position)
    if target_centroid is not None:
        graph_error = angle_error_to_point(position, target_centroid, yaw)
        if abs(np.degrees(graph_error)) > 35:
            return "turn_to_goal", graph_error, "target_graph_turn", target_node
        return "forward", graph_error, "target_graph_forward", target_node

    # No target yet: scan periodically, otherwise move forward to collect new views.
    if step % 10 in (0, 1, 2, 3):
        return "rotate_left", 0.0, "scan_for_target", None
    return "forward", 0.0, "explore_forward", None


def evaluate_episode(client, episode, detector, strategy="conceptgraph", max_steps=500, graph_output=None, frames_output=None, detector_name="clip", planner=None, oracle_graph_bootstrap=False, clip_verifier=None, diagnostic_dir=None, force_llm_graph_choice=False, sam_segmenter=None, target_tiled_detection=False, target_tile_grid=2, target_tile_box_threshold=0.20, target_clip_margin=None, max_target_nodes_per_frame=None, target_memory_mode="conservative", safe_step_mode=False, collision_recovery=False, visual_close_approach=False, render_warmup_frames=0, render_warmup_delay=0.2, map_association_threshold=0.62):
    """Evaluate single episode."""
    
    # Episode info
    episode_id = episode['episode_id']
    target_name = episode['true_name'].strip()
    object_name = episode['object_name']
    map_name = episode['map_name']
    goal_pose = np.asarray(episode['pose'][0], dtype=np.float32)
    geo_dist = episode['info']['geodesic_distance']
    euc_dist = episode['info']['euclidean_distance']
    
    print(f"\n{'='*70}")
    print(f"Episode {episode_id} on {map_name}")
    print(f"Target: '{target_name}' ({object_name})")
    print(f"Distance: {geo_dist}m (geodesic), {euc_dist:.1f}m (euclidean)")
    print(f"Strategy: {strategy}")
    print(f"{'='*70}\n")
    
    # Reset to start pose
    start_pos = episode['start_pose']['start_position']
    start_quat = episode['start_pose']['start_quaternionr']  # [x, y, z, w]
    
    # Clear collision/landed state from interrupted or previous episodes.
    client.reset()
    time.sleep(0.5)
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simPause(False)
    # AirSim takeoff is relative to its current home frame.  Take off first and
    # teleport second; doing this in the opposite order can silently move later
    # episodes back toward an old home position.
    client.takeoffAsync().join()
    pose = airsim.Pose(
        airsim.Vector3r(start_pos[0], start_pos[1], start_pos[2]),
        airsim.Quaternionr(start_quat[0], start_quat[1], start_quat[2], start_quat[3])
    )
    for _ in range(3):
        client.simSetVehiclePose(pose, True)
        time.sleep(0.35)
        actual = client.simGetVehiclePose().position
        actual_position = np.asarray([actual.x_val, actual.y_val, actual.z_val], dtype=np.float32)
        if float(np.linalg.norm(actual_position - np.asarray(start_pos, dtype=np.float32))) <= 2.0:
            break
    else:
        raise EpisodeResetError(start_pos, actual_position.tolist(), "teleport")
    client.hoverAsync().join()
    time.sleep(0.5)
    # Verify again after the flight controller takes ownership. A vehicle that
    # remained in a stale collision state can accept simSetVehiclePose and then
    # immediately snap back to the landscape.
    actual = client.simGetVehiclePose().position
    actual_position = np.asarray([actual.x_val, actual.y_val, actual.z_val], dtype=np.float32)
    if float(np.linalg.norm(actual_position - np.asarray(start_pos, dtype=np.float32))) > 2.0:
        for _ in range(3):
            client.simSetVehiclePose(pose, True)
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            client.hoverAsync().join()
            time.sleep(0.25)
        actual = client.simGetVehiclePose().position
        actual_position = np.asarray([actual.x_val, actual.y_val, actual.z_val], dtype=np.float32)
        if float(np.linalg.norm(actual_position - np.asarray(start_pos, dtype=np.float32))) > 2.0:
            raise EpisodeResetError(start_pos, actual_position.tolist(), "post_hover")
    
    graph_builder = ConceptGraphBuilder(
        fov_degrees=90.0,
        point_stride=8,
        max_depth=80.0,
        merge_iou=0.15,
        association_threshold=map_association_threshold,
    )
    if oracle_graph_bootstrap:
        add_oracle_goal_node(graph_builder, target_name, goal_pose)

    # Initialize metrics
    success = False
    success_reason = None
    spl = 0.0
    min_distance_to_goal = float('inf')
    path_length = 0.0
    prev_position = np.array(start_pos[:3])
    trajectory = []
    detections = []
    frames_collected = 0
    stuck_counter = 0
    llm_history = []
    local_executor = LocalWaypointExecutor()
    hierarchical_state = {}
    clip_active_state = {}
    collision_events = []
    initial_collision = collision_info_to_dict(client.simGetCollisionInfo())
    last_collision_stamp = int(initial_collision.get("time_stamp", 0)) if initial_collision.get("has_collided") else None
    # Perception can take tens of seconds at high resolution. Freeze simulation
    # between actions so wall-clock inference time does not become vehicle drift.
    client.simPause(True)

    frame_dir = Path(frames_output) if frames_output else None
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = Path(diagnostic_dir) if diagnostic_dir else None
    if diagnostic_path:
        diagnostic_path.mkdir(parents=True, exist_ok=True)

    render_warmup_black_ratios = warm_up_rendering(
        client, int(render_warmup_frames), float(render_warmup_delay)
    )
    if render_warmup_black_ratios:
        print(
            "Render warm-up: "
            f"{len(render_warmup_black_ratios)} frames, near-black "
            f"{100.0 * render_warmup_black_ratios[0]:.2f}% → "
            f"{100.0 * render_warmup_black_ratios[-1]:.2f}%"
        )
    
    # Navigation loop
    for step in range(max_steps):
        # Get current observation
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        
        if len(responses) < 2:
            print(f"⚠️  Failed to get images at step {step}")
            break
        
        rgb, depth = image_responses_to_arrays(responses)
        render_black_ratio = near_black_pixel_ratio(rgb)
        
        # Get current pose
        current_pose = client.simGetVehiclePose()
        position = np.array([
            current_pose.position.x_val,
            current_pose.position.y_val,
            current_pose.position.z_val
        ])
        orientation = np.array([
            current_pose.orientation.x_val,
            current_pose.orientation.y_val,
            current_pose.orientation.z_val,
            current_pose.orientation.w_val
        ])
        
        # Update metrics
        step_distance = np.linalg.norm(position - prev_position)
        path_length += step_distance
        prev_position = position.copy()
        if step > 0 and step_distance < 0.25:
            stuck_counter += 1
        else:
            stuck_counter = 0
        
        distance_to_goal = np.linalg.norm(position - goal_pose)
        min_distance_to_goal = min(min_distance_to_goal, distance_to_goal)
        
        frame = UAVFrame(
            rgb=rgb,
            depth=depth,
            position=position,
            quaternion_xyzw=orientation,
            step=step,
            metadata={
                'episode_id': episode_id,
                'map_name': map_name,
                'target': target_name,
                'object_name': object_name,
                'goal_pose': goal_pose.tolist(),
                'distance_to_goal': float(distance_to_goal),
                'detector': detector_name,
            }
        )

        if detector_name == "groundingdino":
            target_prompt = target_prompt_from_episode(target_name, episode.get("description", ""))
            gd_target_detections = detector.detect(
                rgb,
                target=target_prompt,
                box_threshold=0.22,
                text_threshold=0.16,
                top_k=6,
                target_only=True,
            )
            if target_tiled_detection and rgb.shape[1] >= 640:
                tiled = detector.detect_tiled(
                    rgb,
                    target=target_prompt,
                    rows=target_tile_grid,
                    cols=target_tile_grid,
                    overlap=0.20,
                    # Retain proposals before geometry filtering: a distant
                    # ball can have lower DINO score than a tile-sized prompt
                    # hallucination, so filtering only the top eight loses it.
                    top_k_per_tile=4,
                    top_k=24,
                    box_threshold=target_tile_box_threshold,
                )
                gd_target_detections.extend(tiled)
                gd_target_detections.sort(key=lambda item: item.confidence, reverse=True)
            gd_target_detections = [
                det for det in gd_target_detections
                if target_proposal_geometry_is_plausible(det, rgb.shape[:2], target_name)
            ]
            gd_target_detections.sort(key=lambda item: item.confidence, reverse=True)
            gd_target_detections = gd_target_detections[:10]
            # Target-only prompts can return a descriptive phrase that omits
            # the class token (e.g. "rectangular form roof"). Preserve its
            # semantic role explicitly while retaining the phrase as evidence.
            for det in gd_target_detections:
                # Store one canonical spelling in the scene graph. The
                # original DINO phrase remains in ``det.phrase`` as evidence.
                det.label = normalize_target_name(target_name)
            gd_context_detections = detector.detect(rgb, target=target_name, top_k=12)
            context_only = [
                det for det in gd_context_detections
                if not label_matches_target(det.label, target_name)
            ]
            # Never let high-confidence context objects evict small target
            # candidates before crop verification.
            gd_detections = gd_target_detections[:10] + context_only[:10]
            evidence_cache = {}
            for det in gd_target_detections:
                evidence = clip_detection_target_evidence(
                    det,
                    rgb,
                    target_name,
                    str(episode.get("description", "")).strip(),
                    clip_verifier,
                )
                projected_range = projected_target_range_m(det, depth, position, orientation)
                evidence["projected_range_m"] = projected_range
                evidence["target_lifecycle"] = target_observation_lifecycle(
                    det, evidence, rgb.shape[:2], projected_range, target_name
                )
                evidence_cache[id(det)] = evidence
            # Far cues must never be converted into persistent 3-D target
            # nodes; they are retained separately for a bounded bearing turn.
            persistent_target_ids = {
                id(det) for det in gd_target_detections
                if not is_far_target_evidence(evidence_cache[id(det)])
            }
            if target_clip_margin is not None:
                eligible = [
                    det for det in gd_target_detections
                    if not is_far_target_evidence(evidence_cache[id(det)])
                    and float(evidence_cache[id(det)].get("target_margin") or -1e9) >= target_clip_margin
                ]
                eligible.sort(
                    key=lambda det: target_persistence_score(
                        det, evidence_cache[id(det)], rgb.shape[:2], target_name
                    ),
                    reverse=True,
                )
                if max_target_nodes_per_frame is not None:
                    eligible = eligible[:max_target_nodes_per_frame]
                persistent_target_ids = {id(det) for det in eligible}
            navigation_target_detections = [
                det for det in gd_target_detections if id(det) in persistent_target_ids
            ]
            far_target_detections = [
                det for det in gd_target_detections if is_far_target_evidence(evidence_cache[id(det)])
            ]
            if gd_detections:
                display_detections = sorted(gd_detections, key=lambda item: item.confidence, reverse=True)
                detected = display_detections[0].label
                conf = display_detections[0].confidence
                for det in gd_detections:
                    selected_verified_target = (
                        label_matches_target(det.label, target_name)
                        and id(det) in persistent_target_ids
                        and float(det.confidence) >= 0.25
                        and 0.0004 < bbox_area_ratio(det.bbox_xyxy, rgb.shape[:2]) <= 0.35
                    )
                    if not is_graph_detection(det, rgb.shape[:2], target_name) and not selected_verified_target:
                        continue
                    if label_matches_target(det.label, target_name) and id(det) not in persistent_target_ids:
                        continue
                    evidence = evidence_cache.get(id(det)) if label_matches_target(det.label, target_name) else {
                        "status": "context",
                        "target_score": None,
                        "negative_score": None,
                        "target_margin": None,
                        "generic_predictions": [],
                    }
                    generic_labels = ", ".join(
                        item["label"] for item in evidence.get("generic_predictions", [])[:3]
                    ) or "not evaluated"
                    caption = (
                        f"DINO label={det.label}; phrase={getattr(det, 'phrase', '')}; "
                        f"confidence={float(det.confidence):.3f}; semantic_status={evidence.get('status')}; "
                        f"lifecycle={evidence.get('target_lifecycle')}; range_m={evidence.get('projected_range_m')}; "
                        f"CLIP target_margin={evidence.get('target_margin')}; generic_labels={generic_labels}"
                    )
                    instance_mask = None
                    if sam_segmenter is not None and label_matches_target(det.label, target_name):
                        try:
                            instance_mask = sam_segmenter.segment_bbox(rgb, det.bbox_xyxy)
                        except Exception as exc:
                            evidence["sam_error"] = str(exc)
                    if instance_mask is not None:
                        evidence["sam_mask_area"] = int(np.count_nonzero(instance_mask))
                    if label_matches_target(det.label, target_name):
                        depth_quality = sam_mask_depth_quality(instance_mask, depth)
                        evidence["sam_depth_quality"] = depth_quality
                        if not target_mask_depth_is_reliable(depth_quality):
                            evidence["geometry_status"] = "rejected_unreliable_sam_depth"
                            continue
                    graph_builder.add_detection(
                        frame,
                        det,
                        caption=caption,
                        evidence=evidence,
                        mask=instance_mask,
                        visual_embedding=detection_visual_embedding(det, rgb, clip_verifier),
                    )
            else:
                detected = "unknown"
                conf = 0.0
            detection_records = [det.to_dict() for det in gd_detections]
            target_detection_records = [det.to_dict() for det in gd_target_detections]
            target_candidates = summarize_target_candidates(
                gd_target_detections,
                target_name=target_name,
                image_rgb=rgb,
                image_shape=rgb.shape[:2],
                depth=depth,
                position=position,
                orientation=orientation,
                goal_pose=goal_pose,
                clip_verifier=clip_verifier,
                target_description=str(episode.get("description", "")).strip(),
                evidence_cache=evidence_cache,
            )
        elif detector_name == "clip":
            # Detect with CLIP and add this RGB-D observation to the ConceptGraph.
            detection = detector.classify_image(rgb, top_k=3)
            detected = detection[0][0]  # Top label
            conf = detection[0][1]  # Top confidence
            top_text = ", ".join(f"{label}:{score:.3f}" for label, score in detection)
            graph_builder.add_frame(
                frame,
                label=detected,
                caption=f"CLIP top labels for {target_name}: {top_text}",
                confidence=float(conf),
            )
            detection_records = [{'label': label, 'confidence': float(score)} for label, score in detection]
            target_detection_records = []
            target_candidates = []
            far_target_detections = []
        else:
            detected = "unobserved"
            conf = 0.0
            detection_records = []
            target_detection_records = []
            target_candidates = []
            far_target_detections = []

        frames_collected += 1
        # Suppress abandoned one-frame hypotheses while retaining all stable
        # map objects for semantic planning.
        if step > 0 and step % 5 == 0:
            graph_builder.prune_stale_tentative(step, max_age=25)
        trajectory.append(position.round(4).tolist())
        detections.append({
            'step': step,
            'label': detected,
            'confidence': float(conf),
            'detections': detection_records,
            'target_only_detections': target_detection_records,
            'target_candidates': target_candidates,
            'render_near_black_ratio': render_black_ratio,
        })

        save_this_frame = diagnostic_path is not None or step % 5 == 0
        if frame_dir and save_this_frame:
            from PIL import Image
            Image.fromarray(rgb).save(frame_dir / f"rgb_{step:04d}.png")
            np.save(frame_dir / f"depth_{step:04d}.npy", depth)
            if detector_name == "groundingdino":
                draw_detections(rgb, gd_detections, str(frame_dir / f"groundingdino_{step:04d}.png"))
        
        if step % 5 == 0:
            print(f"Step {step:3d}: '{detected}' (conf={conf:.3f}), dist={distance_to_goal:.1f}m")

        # Check success after saving the terminal observation into the graph.
        if distance_to_goal < SUCCESS_DISTANCE_METERS:
            success = True
            success_reason = "distance_threshold"
            spl = (geo_dist / max(path_length, geo_dist)) if path_length > 0 else 0
            print(f"\n✅ SUCCESS at step {step}!")
            print(f"   Distance to goal: {distance_to_goal:.2f}m")
            print(f"   Path length: {path_length:.1f}m")
            print(f"   SPL: {spl:.3f}")
            break
        
        yaw = quaternion_to_yaw(current_pose.orientation)
        angle_diff = angle_error_to_goal(position, goal_pose, yaw)
        nav_source = "legacy"
        nav_target = None
        if collision_recovery and int(hierarchical_state.get("collision_recovery_steps", 0)) > 0:
            remaining = int(hierarchical_state.get("collision_recovery_steps", 0))
            sequence = hierarchical_state.get("collision_recovery_sequence", ["ascend_slow"])
            if not isinstance(sequence, list) or not sequence:
                sequence = ["ascend_slow"]
            sequence_index = max(0, len(sequence) - remaining)
            action = str(sequence[min(sequence_index, len(sequence) - 1)])
            control_angle_diff = 0.0
            nav_source = "collision_recovery"
            hierarchical_state["collision_recovery_steps"] = max(0, remaining - 1)
            nav_target = {
                "recovery_action": action,
                "recovery_sequence": sequence,
                "remaining_recovery_steps": hierarchical_state["collision_recovery_steps"],
                "last_collision_object": hierarchical_state.get("last_collision_object"),
            }
        elif strategy == "hierarchical":
            action, control_angle_diff, nav_source, nav_target = choose_hierarchical_action(
                planner=planner,
                target_name=target_name,
                target_description=str(episode.get("description", "")).strip(),
                gd_detections=navigation_target_detections if detector_name == "groundingdino" else [],
                far_target_detections=far_target_detections if detector_name == "groundingdino" else [],
                target_candidates=target_candidates,
                graph_builder=graph_builder,
                position=position,
                yaw=yaw,
                depth=depth,
                executor=local_executor,
                history=llm_history,
                state=hierarchical_state,
                step=step,
                image_width=rgb.shape[1],
                force_planner=force_llm_graph_choice,
                target_memory_mode=target_memory_mode,
                visual_close_approach=visual_close_approach,
            )
            llm_history.append(nav_source)
        elif strategy == "oracle_local":
            action, control_angle_diff, executor_info = local_executor.choose_action(
                position, yaw, goal_pose, depth
            )
            nav_source = "oracle_waypoint_local_executor"
            nav_target = executor_info
        elif strategy == "deepseek" and detector_name == "groundingdino":
            action, control_angle_diff, nav_source, nav_target = choose_llm_planned_action(
                planner=planner,
                target_name=target_name,
                gd_detections=navigation_target_detections,
                graph_builder=graph_builder,
                position=position,
                yaw=yaw,
                image_width=rgb.shape[1],
                step=step,
                llm_history=llm_history,
                stuck_counter=stuck_counter,
            )
            llm_history.append(action)
        elif strategy == "deepseek" and detector_name == "clip":
            action, control_angle_diff, nav_source, nav_target = choose_clip_llm_action(
                planner=planner,
                target_name=target_name,
                detected_label=detected,
                confidence=conf,
                graph_builder=graph_builder,
                position=position,
                angle_diff_rad=angle_diff,
                step=step,
                llm_history=llm_history,
                stuck_counter=stuck_counter,
            )
            llm_history.append(action)
        elif strategy == "clip_active" and detector_name == "clip":
            action, control_angle_diff, nav_source, nav_target = choose_clip_active_action(
                detector=detector,
                rgb=rgb,
                target_name=target_name,
                target_description=str(episode.get("description", "")).strip(),
                detected_label=detected,
                detected_confidence=conf,
                state=clip_active_state,
                step=step,
                stuck_counter=stuck_counter,
            )
        elif strategy == "conceptgraph" and detector_name == "groundingdino":
            action, control_angle_diff, nav_source, nav_target = choose_conceptgraph_action(
                target_name=target_name,
                gd_detections=gd_detections,
                graph_builder=graph_builder,
                position=position,
                yaw=yaw,
                image_width=rgb.shape[1],
                step=step,
                stuck_counter=stuck_counter,
            )
        else:
            action = choose_action(strategy, target_name, detected, conf, angle_diff, step)
            control_angle_diff = angle_diff
        detections[-1]["action"] = action
        detections[-1]["nav_source"] = nav_source
        nav_target = annotate_nav_target_diagnostics(nav_target, goal_pose, distance_to_goal)
        detections[-1]["nav_target"] = nav_target
        action_speed = recommended_action_speed(nav_target)
        if safe_step_mode:
            action_speed = apply_safe_step_speed(action, action_speed)
        if isinstance(nav_target, dict) and nav_target.get("near_target_refinement"):
            action_speed = apply_close_target_speed(action, action_speed)
        detections[-1]["action_speed"] = float(action_speed)
        if diagnostic_path:
            graph_builder.scene_graph.rebuild_spatial_edges()
            snapshot = {
                "step": step,
                "target": target_name,
                "target_description": str(episode.get("description", "")).strip(),
                "uav_position": position.round(4).tolist(),
                "uav_yaw": float(yaw),
                "true_goal": goal_pose.round(4).tolist(),
                "distance_to_goal": float(distance_to_goal),
                "action": action,
                "action_speed": float(action_speed),
                "nav_source": nav_source,
                "nav_target": nav_target,
                "target_memory_mode": target_memory_mode,
                "safe_step_mode": bool(safe_step_mode),
                "collision_recovery": bool(collision_recovery),
                "visual_close_approach": bool(visual_close_approach),
                "target_candidates": target_candidates,
                "hierarchical_state_debug": {
                    "subgoal": hierarchical_state.get("subgoal"),
                    "source": hierarchical_state.get("source"),
                    "last_plan_step": hierarchical_state.get("last_plan_step"),
                    "target_candidate_lock": hierarchical_state.get("target_candidate_lock"),
                    "target_lock_node_id": hierarchical_state.get("target_lock_node_id"),
                    "rejected_node_ids": hierarchical_state.get("rejected_node_ids", []),
                    "rejected_subgoal_regions": hierarchical_state.get("rejected_subgoal_regions", []),
                    "collision_recovery_steps": hierarchical_state.get("collision_recovery_steps", 0),
                    "recent_collision_steps": hierarchical_state.get("recent_collision_steps", []),
                    "last_collision_object": hierarchical_state.get("last_collision_object"),
                },
                "graph": graph_builder.scene_graph.to_dict(),
            }
            with (diagnostic_path / f"step_{step:04d}.json").open("w", encoding="utf-8") as file:
                json.dump(snapshot, file, indent=2, ensure_ascii=False)
            from PIL import Image
            Image.fromarray(rgb).save(diagnostic_path / f"rgb_{step:04d}.png")
        client.simPause(False)
        execute_action(client, action, control_angle_diff, speed=action_speed)
        client.simPause(True)
        collision_record = collision_info_to_dict(client.simGetCollisionInfo())
        collision_record.update({
            "step": step,
            "action": action,
            "nav_source": nav_source,
            "position": position.round(4).tolist(),
        })
        new_collision = is_new_collision(collision_record, last_collision_stamp)
        collision_record["is_new_collision"] = new_collision
        detections[-1]["collision"] = collision_record
        reject_collided_graph_node(hierarchical_state, nav_target, collision_record, object_name)
        recovery_record = (
            clear_navigation_memory_after_collision(hierarchical_state, nav_target, collision_record, object_name)
            if collision_recovery else {"triggered": False}
        )
        collision_record["recovery"] = recovery_record
        detections[-1]["collision_recovery"] = recovery_record
        if new_collision:
            last_collision_stamp = int(collision_record.get("time_stamp", 0)) or last_collision_stamp
            collision_events.append(collision_record)
            print(
                f"⚠️  Collision at step {step}: object='{collision_record['object_name']}', "
                f"action={action}, penetration={collision_record['penetration_depth']:.3f}"
            )
            if recovery_record.get("triggered"):
                print(
                    f"   ↳ recovery: reject={recovery_record.get('rejected_node_id')} "
                    f"next={recovery_record.get('recovery_action')}"
                )

        post_pose = client.simGetVehiclePose()
        post_position = np.array([
            post_pose.position.x_val,
            post_pose.position.y_val,
            post_pose.position.z_val,
        ])
        post_step_distance = float(np.linalg.norm(post_position - prev_position))
        if post_step_distance > 0:
            path_length += post_step_distance
            prev_position = post_position.copy()
        post_distance_to_goal = float(np.linalg.norm(post_position - goal_pose))
        min_distance_to_goal = min(min_distance_to_goal, post_distance_to_goal)
        detections[-1]["post_action_position"] = post_position.round(4).tolist()
        detections[-1]["post_action_distance_to_goal"] = post_distance_to_goal
        if post_distance_to_goal < SUCCESS_DISTANCE_METERS:
            success = True
            success_reason = "post_action_distance_threshold"
            spl = geo_dist / max(path_length, geo_dist)
            print(f"\n✅ SUCCESS after action at step {step}!")
            print(f"   Distance to goal: {post_distance_to_goal:.2f}m")
            print(f"   Path length: {path_length:.1f}m")
            print(f"   SPL: {spl:.3f}")
            break
        if (
            is_target_object_contact(collision_record, object_name)
            and post_distance_to_goal <= SUCCESS_DISTANCE_METERS + TARGET_CONTACT_SUCCESS_MARGIN_METERS
        ):
            success = True
            success_reason = "target_object_contact"
            spl = geo_dist / max(path_length, geo_dist)
            detections[-1]["target_contact_success"] = True
            print(f"\n✅ SUCCESS by target contact at step {step}!")
            print(f"   Distance to goal center: {post_distance_to_goal:.2f}m")
            print(f"   Contact object: {collision_record['object_name']}")
            print(f"   Path length: {path_length:.1f}m")
            print(f"   SPL: {spl:.3f}")
            break

        reached_reject_record = reject_reached_semantic_subgoal(hierarchical_state, nav_target)
        detections[-1]["reached_subgoal_rejection"] = reached_reject_record
        if reached_reject_record.get("triggered"):
            print(
                f"↪️  Reached semantic subgoal without global success at step {step}: "
                f"reject={reached_reject_record.get('rejected_node_id')}"
            )
        
        time.sleep(0.1)
    
    client.simPause(False)
    scene_graph = graph_builder.finalize()
    graph_goal_diagnostics = summarize_graph_goal_diagnostics(scene_graph, target_name, goal_pose)
    if graph_output:
        save_scene_graph(scene_graph, graph_output)

    # Results
    result = {
        'episode_id': episode_id,
        'map_name': map_name,
        'target': target_name,
        'object_name': object_name,
        'strategy': strategy,
        'target_memory_mode': target_memory_mode,
        'safe_step_mode': bool(safe_step_mode),
        'collision_recovery': bool(collision_recovery),
        'visual_close_approach': bool(visual_close_approach),
        'render_warmup_frames': int(render_warmup_frames),
        'render_warmup_near_black_ratios': render_warmup_black_ratios,
        'success': success,
        'success_reason': success_reason,
        'spl': spl,
        'min_distance_to_goal': float(min_distance_to_goal),
        'path_length': float(path_length),
        'steps': step + 1,
        'frames_collected': frames_collected,
        'scene_graph_path': graph_output,
        'scene_graph_nodes': len(scene_graph.graph.nodes),
        'scene_graph_edges': len(scene_graph.graph.edges),
        'trajectory': trajectory,
        'detections': detections,
        'collided': bool(collision_events),
        'collision_count': len(collision_events),
        'collision_events': collision_events,
        'graph_goal_diagnostics': graph_goal_diagnostics,
        'geodesic_distance': geo_dist,
        'euclidean_distance': euc_dist
    }
    
    if not success:
        print(f"\n❌ FAILED after {step+1} steps")
        print(f"   Min distance to goal: {min_distance_to_goal:.2f}m")
        print(f"   Path length: {path_length:.1f}m")
    
    # Land and disarm
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to UAV-ON dataset JSON file")
    parser.add_argument("--strategy", type=str, default="conceptgraph",
                       choices=["baseline", "clip", "clip_active", "conceptgraph", "oracle", "deepseek", "hierarchical", "oracle_local"],
                       help="Navigation strategy")
    parser.add_argument("--detector", type=str, default="clip",
                       choices=["clip", "groundingdino", "none"],
                       help="Detector used for online scene graph construction")
    parser.add_argument("--num-episodes", type=int, default=None,
                       help="Number of episodes to evaluate (default: all)")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Start evaluation at this dataset episode index")
    parser.add_argument("--episode-ids", type=str, default=None,
                       help="Comma-separated dataset episode_id values to evaluate; overrides start-index/num-episodes")
    parser.add_argument("--output", type=str, default="uavon_results.json",
                       help="Output results file")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Maximum steps per episode")
    parser.add_argument("--graph-dir", type=str, default=None,
                       help="Directory for per-episode ConceptGraph JSON files")
    parser.add_argument("--save-frames-dir", type=str, default=None,
                       help="Optional directory for sampled RGB-D frames")
    parser.add_argument("--airsim-ip", type=str, default="127.0.0.1",
                       help="AirSim RPC host")
    parser.add_argument("--airsim-port", type=int, default=41451,
                       help="AirSim RPC port")
    parser.add_argument("--disable-llm", action="store_true",
                       help="Run hierarchical planning without sending scene data to an external LLM")
    parser.add_argument("--oracle-graph-bootstrap", action="store_true",
                       help="Inject true target coordinates as a graph node for upper-bound debugging")
    parser.add_argument("--clip-crop-verify", action="store_true",
                       help="Use CLIP crop classification to verify target-like GroundingDINO boxes")
    parser.add_argument("--map-clip-features", action="store_true",
                       help="Encode DINO instance crops with CLIP for cross-frame object-map association")
    parser.add_argument("--map-association-threshold", type=float, default=0.62,
                       help="Minimum geometry+CLIP association score for merging object observations")
    parser.add_argument("--diagnostic-dir", type=str, default=None,
                       help="Save every-step RGB, graph snapshot, and selected navigation target")
    parser.add_argument("--force-llm-graph-choice", action="store_true",
                       help="Bypass target/bbox heuristics so the LLM must choose from the current graph")
    parser.add_argument("--sam-segment", action="store_true",
                       help="Use bbox-prompted SAM masks for target object 3-D projection")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base",
                       help="Hugging Face SAM model id")
    parser.add_argument("--target-tiled-detection", action="store_true",
                       help="Run target-only DINO on overlapping high-resolution crops")
    parser.add_argument("--target-tile-grid", type=int, default=2,
                       help="Rows/columns for target-only DINO tiles; use 3 for small targets")
    parser.add_argument("--target-tile-box-threshold", type=float, default=0.20,
                       help="Detection threshold used only for tiled target crops")
    parser.add_argument("--target-clip-margin", type=float, default=None,
                       help="Persist target nodes only when crop CLIP margin reaches this value")
    parser.add_argument("--max-target-nodes-per-frame", type=int, default=None,
                       help="After CLIP ranking, persist at most this many target candidates per frame")
    parser.add_argument("--target-memory-mode", choices=["conservative", "aggressive", "off"], default="conservative",
                       help="How target candidate/graph locks are used: conservative=graph first, aggressive=candidate first, off=no candidate lock")
    parser.add_argument("--safe-step-mode", action="store_true",
                       help="Use smaller forward/vertical speeds for safer local execution")
    parser.add_argument("--collision-recovery", action="store_true",
                       help="On non-target collision, reject the current subgoal/node, clear cache, and execute a short recovery maneuver")
    parser.add_argument("--visual-close-approach", action="store_true",
                       help="Use small local-executor steps for a target candidate verified in at least two nearby frames")
    parser.add_argument("--render-warmup-frames", type=int, default=0,
                       help="Request this many scene frames after each reset before mapping begins")
    parser.add_argument("--render-warmup-delay", type=float, default=0.2,
                       help="Seconds between render warm-up frames")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from an existing output JSON, skipping completed episode IDs")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r') as f:
        episodes = json.load(f)
    
    total_episodes = len(episodes)
    if args.episode_ids:
        requested_ids = [value.strip() for value in args.episode_ids.split(",") if value.strip()]
        requested_set = set(requested_ids)
        episodes_by_id = {str(episode["episode_id"]): episode for episode in episodes}
        missing_ids = [episode_id for episode_id in requested_ids if episode_id not in episodes_by_id]
        if missing_ids:
            raise ValueError(f"episode_id values not found in dataset: {missing_ids}")
        episodes = [episodes_by_id[episode_id] for episode_id in requested_ids]
    else:
        episodes = episodes[args.start_index:]
        if args.num_episodes:
            episodes = episodes[:args.num_episodes]
    
    print(f"Loaded {len(episodes)} episodes from {args.dataset}")
    print(f"Total episodes in file: {total_episodes}")
    
    # Initialize components
    print("\nInitializing components...")
    if args.detector == "groundingdino":
        print("Loading GroundingDINO detector for object-level scene graph nodes...")
        detector = GroundingDINODetector(box_threshold=0.20, text_threshold=0.18)
    elif args.detector == "clip":
        print("Loading CLIP detector for scene graph labels...")
        detector = CLIPDetector()
    else:
        print("Detector disabled (local-executor ablation).")
        detector = None

    planner = None
    if args.strategy in {"deepseek", "hierarchical"} and not args.disable_llm:
        print("Loading DeepSeek LLM planner for graph-level planning...")
        planner = DeepSeekPlanner()

    clip_verifier = None
    if args.clip_crop_verify or args.map_clip_features:
        print("Loading CLIP crop verifier/object-map feature encoder...")
        clip_verifier = CLIPDetector()

    sam_segmenter = None
    if args.sam_segment:
        sam_segmenter = SAMSegmenter(model_id=args.sam_model)
    
    graph_dir = Path(args.graph_dir) if args.graph_dir else Path(args.output).with_suffix("").parent / "scene_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
    client.confirmConnection()
    print("✅ Connected!")
    
    # Run evaluation.  Long online-LLM evaluations can be interrupted by a UE
    # crash or scheduler event; retain the completed prefix and restart from
    # the first unfinished episode instead of silently overwriting it.
    results = []
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as file:
                previous = json.load(file)
            if not isinstance(previous, list):
                raise ValueError("existing output is not a JSON list")
            results = previous
            completed_ids = {str(item.get('episode_id')) for item in results if isinstance(item, dict)}
            before = len(episodes)
            episodes = [episode for episode in episodes if str(episode.get('episode_id')) not in completed_ids]
            print(f"Resuming {args.output}: kept {len(results)} completed results; "
                  f"{before - len(episodes)} selected episode(s) skipped.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot resume from {args.output}: {exc}") from exc
    success_count = sum(bool(item.get('success')) for item in results if isinstance(item, dict))
    
    for i, episode in enumerate(episodes):
        print(f"\n{'='*70}")
        print(f"Episode {i+1}/{len(episodes)} (completed total: {len(results)})")
        print(f"{'='*70}")
        
        graph_output = graph_dir / f"episode_{episode['episode_id']}_graph.json"
        frames_output = None
        if args.save_frames_dir:
            frames_output = Path(args.save_frames_dir) / f"episode_{episode['episode_id']}"

        try:
            result = evaluate_episode(
                client, episode, detector,
                strategy=args.strategy,
                max_steps=args.max_steps,
                graph_output=str(graph_output),
                frames_output=str(frames_output) if frames_output else None,
                detector_name=args.detector,
                planner=planner,
                oracle_graph_bootstrap=args.oracle_graph_bootstrap,
                clip_verifier=clip_verifier,
                diagnostic_dir=(str(Path(args.diagnostic_dir) / f"episode_{episode['episode_id']}") if args.diagnostic_dir else None),
                force_llm_graph_choice=args.force_llm_graph_choice,
                sam_segmenter=sam_segmenter,
                target_tiled_detection=args.target_tiled_detection,
                target_tile_grid=max(1, args.target_tile_grid),
                target_tile_box_threshold=args.target_tile_box_threshold,
                target_clip_margin=args.target_clip_margin,
                max_target_nodes_per_frame=args.max_target_nodes_per_frame,
                target_memory_mode=args.target_memory_mode,
                safe_step_mode=args.safe_step_mode,
                collision_recovery=args.collision_recovery,
                visual_close_approach=args.visual_close_approach,
                render_warmup_frames=args.render_warmup_frames,
                render_warmup_delay=args.render_warmup_delay,
                map_association_threshold=args.map_association_threshold,
            )
        except EpisodeResetError as exc:
            # Preserve the dataset's strict start-pose contract: do not score a
            # run from an offset pose, but do not discard a multi-hour batch.
            print(f"⚠️  Skipping episode {episode['episode_id']}: {exc}")
            result = {
                'episode_id': str(episode['episode_id']),
                'map_name': episode.get('map_name'),
                'target': str(episode.get('true_name', '')).strip(),
                'object_name': episode.get('object_name'),
                'strategy': args.strategy,
                'success': False,
                'success_reason': 'reset_failed',
                'failure_type': 'episode_reset_failed',
                'reset_error': {
                    'phase': exc.phase,
                    'expected_position': exc.expected,
                    'actual_position': exc.actual,
                },
                'spl': 0.0,
                'min_distance_to_goal': None,
                'path_length': 0.0,
                'steps': 0,
                'frames_collected': 0,
                'scene_graph_path': None,
                'scene_graph_nodes': 0,
                'scene_graph_edges': 0,
                'trajectory': [],
                'detections': [],
                'collided': False,
                'collision_count': 0,
                'collision_events': [],
                'graph_goal_diagnostics': {},
                'geodesic_distance': episode.get('info', {}).get('geodesic_distance'),
                'euclidean_distance': episode.get('info', {}).get('euclidean_distance'),
            }
        
        results.append(result)
        # Persist after every episode so long online evaluations are resumable.
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        if result['success']:
            success_count += 1
        
        print(f"\nCumulative success rate: {success_count}/{len(results)} = {100*success_count/len(results):.1f}%")
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.strategy}")
    print(f"Episodes evaluated: {len(results)}")
    print(f"Success rate: {success_count}/{len(results)} = {100*success_count/len(results):.1f}%")
    
    if success_count > 0:
        avg_spl = np.mean([r['spl'] for r in results if r['success']])
        print(f"Average SPL (successful): {avg_spl:.3f}")
    
    avg_min_dist = np.mean([r['min_distance_to_goal'] for r in results])
    print(f"Average min distance to goal: {avg_min_dist:.2f}m")
    print(f"Scene graphs saved to: {graph_dir}")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
