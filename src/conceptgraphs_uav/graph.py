from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .frame import UAVFrame
from .geometry import axis_aligned_bbox, bbox_iou_3d, project_depth_bbox_to_world, project_depth_mask_to_world, project_depth_to_world


@dataclass
class ConceptNode:
    node_id: str
    label: str
    caption: str
    centroid: List[float]
    bbox_min: List[float]
    bbox_max: List[float]
    observations: int = 1
    confidence: float = 1.0
    visual_embedding: List[float] = field(default_factory=list)
    state: str = "tentative"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneGraph:
    def __init__(self) -> None:
        self.graph = nx.Graph()

    def add_or_update_node(
        self,
        node: ConceptNode,
        merge_iou: float = 0.15,
        association_threshold: float = 0.62,
    ) -> str:
        match_id, association = self._find_match(
            node, merge_iou=merge_iou, association_threshold=association_threshold
        )
        if match_id is None:
            self.graph.add_node(node.node_id, **asdict(node))
            return node.node_id

        current = self.graph.nodes[match_id]
        count = int(current["observations"])
        new_count = count + 1
        current["centroid"] = _weighted_average(current["centroid"], node.centroid, count)
        current["bbox_min"] = np.minimum(current["bbox_min"], node.bbox_min).tolist()
        current["bbox_max"] = np.maximum(current["bbox_max"], node.bbox_max).tolist()
        current["observations"] = new_count
        current["confidence"] = max(float(current["confidence"]), float(node.confidence))
        current["state"] = "confirmed" if new_count >= 3 else "tentative"
        current["last_seen_step"] = int(node.metadata.get("step", -1))
        current["association_score"] = round(float(association), 4)
        current["visual_embedding"] = _merge_embeddings(
            current.get("visual_embedding", []), node.visual_embedding, count
        )
        incoming_metadata = node.metadata if isinstance(node.metadata, dict) else {}
        incoming_evidence = incoming_metadata.get("semantic_evidence", {})
        if isinstance(incoming_evidence, dict) and incoming_evidence.get("target_lifecycle"):
            current_metadata = current.setdefault("metadata", {})
            current_evidence = current_metadata.get("semantic_evidence", {})
            merged_evidence = dict(current_evidence) if isinstance(current_evidence, dict) else {}
            merged_evidence.update(incoming_evidence)
            # A close provisional observation becomes navigable only after it
            # has been geometrically merged across three observations.
            if (
                merged_evidence.get("target_lifecycle") == "provisional"
                and new_count >= 3
            ):
                merged_evidence["target_lifecycle"] = "verified"
            current_metadata["semantic_evidence"] = merged_evidence
        if node.caption and node.caption not in current.get("caption", ""):
            current["caption"] = f"{current['caption']} | {node.caption}" if current.get("caption") else node.caption
        return match_id

    def prune_stale_tentative(self, current_step: int, max_age: int = 25) -> List[str]:
        """Remove old single-view hypotheses; confirmed objects remain mapped."""
        removed = []
        for node_id, node in list(self.graph.nodes(data=True)):
            last_seen = int(node.get("last_seen_step", node.get("metadata", {}).get("step", current_step)))
            if node.get("state", "tentative") == "tentative" and current_step - last_seen > max_age:
                self.graph.remove_node(node_id)
                removed.append(str(node_id))
        return removed

    def rebuild_spatial_edges(self, near_distance: float = 12.0) -> None:
        self.graph.remove_edges_from(list(self.graph.edges()))
        nodes = list(self.graph.nodes(data=True))
        for index, (src_id, src) in enumerate(nodes):
            src_center = np.asarray(src["centroid"], dtype=np.float32)
            for dst_id, dst in nodes[index + 1 :]:
                dst_center = np.asarray(dst["centroid"], dtype=np.float32)
                distance = float(np.linalg.norm(src_center - dst_center))
                if distance <= near_distance:
                    delta = dst_center - src_center
                    if abs(float(delta[2])) > max(abs(float(delta[0])), abs(float(delta[1]))):
                        direction = "above" if delta[2] < 0 else "below"
                    elif abs(float(delta[0])) >= abs(float(delta[1])):
                        direction = "in_front_of" if delta[0] > 0 else "behind"
                    else:
                        direction = "right_of" if delta[1] > 0 else "left_of"
                    self.graph.add_edge(src_id, dst_id, relation="near", directional_relation=direction, distance=distance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [dict(data) for _, data in self.graph.nodes(data=True)],
            "edges": [
                {"source": src, "target": dst, **dict(data)}
                for src, dst, data in self.graph.edges(data=True)
            ],
        }

    def _find_match(
        self, node: ConceptNode, merge_iou: float, association_threshold: float
    ) -> Tuple[Optional[str], float]:
        node_min = np.asarray(node.bbox_min, dtype=np.float32)
        node_max = np.asarray(node.bbox_max, dtype=np.float32)
        node_center = np.asarray(node.centroid, dtype=np.float32)
        best_id, best_score = None, -1.0
        for node_id, current in self.graph.nodes(data=True):
            if not _labels_compatible(str(current.get("label", "")), node.label):
                continue
            cur_min = np.asarray(current["bbox_min"], dtype=np.float32)
            cur_max = np.asarray(current["bbox_max"], dtype=np.float32)
            cur_center = np.asarray(current["centroid"], dtype=np.float32)
            center_distance = float(np.linalg.norm(cur_center - node_center))
            iou = bbox_iou_3d(cur_min, cur_max, node_min, node_max)
            # A visual match never overrides a wildly inconsistent 3-D pose.
            if center_distance > 12.0 and iou < merge_iou:
                continue
            spatial = max(iou, float(np.exp(-center_distance / 5.0)))
            visual = _cosine_similarity(current.get("visual_embedding", []), node.visual_embedding)
            # Labels are already gated above; visual evidence becomes important
            # when a detector box contains variable amounts of background.
            score = 0.50 * spatial + 0.40 * visual + 0.10
            if score > best_score:
                best_id, best_score = str(node_id), float(score)
        if best_score >= association_threshold:
            return best_id, best_score
        return None, best_score


class ConceptGraphBuilder:
    def __init__(
        self,
        fov_degrees: float = 90.0,
        point_stride: int = 8,
        max_depth: float = 80.0,
        merge_iou: float = 0.15,
        association_threshold: float = 0.62,
    ) -> None:
        self.fov_degrees = fov_degrees
        self.point_stride = point_stride
        self.max_depth = max_depth
        self.merge_iou = merge_iou
        self.association_threshold = association_threshold
        self.scene_graph = SceneGraph()
        self._next_node_index = 0

    def add_frame(
        self,
        frame: UAVFrame,
        label: Optional[str] = None,
        caption: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        points = project_depth_to_world(
            depth=frame.depth,
            position=frame.position,
            quaternion_xyzw=frame.quaternion_xyzw,
            fov_degrees=self.fov_degrees,
            stride=self.point_stride,
            max_depth=self.max_depth,
        )
        bbox_min, bbox_max = axis_aligned_bbox(points)
        centroid = points.mean(axis=0) if len(points) else frame.position
        node = ConceptNode(
            node_id=f"node_{self._next_node_index:06d}",
            label=label or frame.metadata.get("object_name") or "unknown",
            caption=caption or frame.metadata.get("description") or "",
            centroid=np.asarray(centroid, dtype=np.float32).round(4).tolist(),
            bbox_min=np.asarray(bbox_min, dtype=np.float32).round(4).tolist(),
            bbox_max=np.asarray(bbox_max, dtype=np.float32).round(4).tolist(),
            confidence=float(confidence),
            metadata={"step": frame.step, **frame.metadata},
        )
        self._next_node_index += 1
        return self.scene_graph.add_or_update_node(
            node, merge_iou=self.merge_iou, association_threshold=self.association_threshold
        )

    def add_detection(
        self,
        frame: UAVFrame,
        detection: Any,
        caption: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
        mask: Optional[np.ndarray] = None,
        visual_embedding: Optional[Sequence[float]] = None,
    ) -> Optional[str]:
        """Add one object-level detection to the scene graph using bbox depth."""
        if hasattr(detection, "bbox_xyxy"):
            bbox_xyxy = detection.bbox_xyxy
            label = detection.label
            confidence = float(detection.confidence)
            phrase = getattr(detection, "phrase", "")
        else:
            bbox_xyxy = detection.get("bbox_xyxy") or detection.get("bbox")
            label = detection.get("label", "object")
            confidence = float(detection.get("confidence", 1.0))
            phrase = detection.get("phrase", "")

        if mask is not None:
            points = project_depth_mask_to_world(
                depth=frame.depth,
                mask=np.asarray(mask, dtype=bool),
                position=frame.position,
                quaternion_xyzw=frame.quaternion_xyzw,
                fov_degrees=self.fov_degrees,
                stride=max(1, self.point_stride // 4),
                max_depth=self.max_depth,
            )
        else:
            points = project_depth_bbox_to_world(
                depth=frame.depth,
                bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float32),
                position=frame.position,
                quaternion_xyzw=frame.quaternion_xyzw,
                fov_degrees=self.fov_degrees,
                stride=max(1, self.point_stride // 4),
                max_depth=self.max_depth,
            )
        if len(points) == 0:
            return None

        bbox_min, bbox_max = axis_aligned_bbox(points)
        centroid = np.median(points, axis=0)
        node = ConceptNode(
            node_id=f"node_{self._next_node_index:06d}",
            label=label,
            caption=caption or phrase or frame.metadata.get("description") or "",
            centroid=np.asarray(centroid, dtype=np.float32).round(4).tolist(),
            bbox_min=np.asarray(bbox_min, dtype=np.float32).round(4).tolist(),
            bbox_max=np.asarray(bbox_max, dtype=np.float32).round(4).tolist(),
            confidence=confidence,
            visual_embedding=_normalise_embedding(visual_embedding),
            state="tentative",
            metadata={
                "step": frame.step,
                "bbox_2d": [float(v) for v in bbox_xyxy],
                "detector": "groundingdino",
                "phrase": phrase,
                "semantic_evidence": evidence or {},
                "geometry_source": "sam_mask" if mask is not None else "bbox_depth",
                "mask_area": int(np.count_nonzero(mask)) if mask is not None else None,
                **frame.metadata,
            },
        )
        self._next_node_index += 1
        return self.scene_graph.add_or_update_node(
            node, merge_iou=self.merge_iou, association_threshold=self.association_threshold
        )

    def prune_stale_tentative(self, current_step: int, max_age: int = 25) -> List[str]:
        return self.scene_graph.prune_stale_tentative(current_step, max_age=max_age)

    def finalize(self, near_distance: float = 12.0) -> SceneGraph:
        self.scene_graph.rebuild_spatial_edges(near_distance=near_distance)
        return self.scene_graph


def _weighted_average(old_value: List[float], new_value: List[float], old_count: int) -> List[float]:
    old = np.asarray(old_value, dtype=np.float32)
    new = np.asarray(new_value, dtype=np.float32)
    return ((old * old_count + new) / (old_count + 1)).round(4).tolist()


def _normalise_embedding(values: Optional[Sequence[float]]) -> List[float]:
    if values is None:
        return []
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32).tolist() if norm > 1e-8 else []


def _merge_embeddings(old_values: Sequence[float], new_values: Sequence[float], old_count: int) -> List[float]:
    if not old_values:
        return _normalise_embedding(new_values)
    if not new_values:
        return _normalise_embedding(old_values)
    old = np.asarray(old_values, dtype=np.float32)
    new = np.asarray(new_values, dtype=np.float32)
    if old.shape != new.shape:
        return _normalise_embedding(old)
    return _normalise_embedding((old * old_count + new) / (old_count + 1))


def _cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if not first or not second:
        # Geometry can still associate observations before CLIP is enabled.
        return 0.5
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def _labels_compatible(first: str, second: str) -> bool:
    first, second = first.strip().lower(), second.strip().lower()
    if first == second:
        return True
    aliases = {
        "bus stop": {"bus shelter", "bus station"},
        "traffic light": {"traffic signal", "stoplight"},
        "caravan": {"camper trailer", "travel trailer", "mobile home trailer"},
    }
    return second in aliases.get(first, set()) or first in aliases.get(second, set())
