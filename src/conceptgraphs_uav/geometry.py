import math
from typing import Tuple

import numpy as np


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion.astype(np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def project_depth_to_world(
    depth: np.ndarray,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    fov_degrees: float = 90.0,
    stride: int = 8,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Project a depth map into sparse world points using a pinhole camera model."""

    if depth.ndim != 2:
        raise ValueError("Depth image must be a 2-D array")

    height, width = depth.shape
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.1) & (z < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]

    # AirSim uses an NED camera frame: X forward, Y right, Z down.  Image rows
    # increase downward, so the third component must have the same sign as
    # (y-cy).  The previous negative sign mirrored geometry vertically.
    camera_points = np.stack(
        [z, (xs - cx) * z / focal, (ys - cy) * z / focal],
        axis=1,
    )
    rotation = quaternion_xyzw_to_rotation(quaternion_xyzw)
    return camera_points @ rotation.T + position.astype(np.float32)


def project_depth_bbox_to_world(
    depth: np.ndarray,
    bbox_xyxy: np.ndarray,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    fov_degrees: float = 90.0,
    stride: int = 2,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Project depth samples inside a 2-D bbox into sparse world points."""

    if depth.ndim != 2:
        raise ValueError("Depth image must be a 2-D array")

    height, width = depth.shape
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32)
    x1 = int(np.floor(np.clip(x1, 0, width - 1)))
    x2 = int(np.ceil(np.clip(x2, 0, width - 1)))
    y1 = int(np.floor(np.clip(y1, 0, height - 1)))
    y2 = int(np.ceil(np.clip(y2, 0, height - 1)))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=np.float32)

    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    ys, xs = np.mgrid[y1:y2:stride, x1:x2:stride]
    z = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.1) & (z < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]

    # GroundingDINO boxes often include large background regions. For object
    # localization, keep the nearer foreground band inside the 2-D box instead
    # of averaging background depth into the object centroid.
    if z.size >= 16:
        near_depth = float(np.percentile(z, 25))
        depth_spread = max(1.5, 0.20 * near_depth)
        foreground = z <= near_depth + depth_spread
        if int(np.count_nonzero(foreground)) >= 8:
            xs = xs[foreground]
            ys = ys[foreground]
            z = z[foreground]

    camera_points = np.stack(
        [z, (xs - cx) * z / focal, (ys - cy) * z / focal],
        axis=1,
    )
    rotation = quaternion_xyzw_to_rotation(quaternion_xyzw)
    return camera_points @ rotation.T + position.astype(np.float32)


def project_depth_mask_to_world(
    depth: np.ndarray,
    mask: np.ndarray,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    fov_degrees: float = 90.0,
    stride: int = 2,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Project only instance-mask depth pixels into world coordinates."""
    if depth.ndim != 2 or mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("Depth and mask must be same-shaped 2-D arrays")
    height, width = depth.shape
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    selected = mask[ys, xs].astype(bool)
    z = depth[ys, xs].astype(np.float32)
    valid = selected & np.isfinite(z) & (z > 0.1) & (z < max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]
    # Trim extreme depth tails while preserving the segmented object's extent.
    if z.size >= 20:
        low, high = np.percentile(z, [5, 90])
        keep = (z >= low) & (z <= high)
        xs, ys, z = xs[keep], ys[keep], z[keep]
    camera_points = np.stack([z, (xs - cx) * z / focal, (ys - cy) * z / focal], axis=1)
    rotation = quaternion_xyzw_to_rotation(quaternion_xyzw)
    return camera_points @ rotation.T + position.astype(np.float32)


def axis_aligned_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        empty = np.zeros(3, dtype=np.float32)
        return empty, empty
    return points.min(axis=0), points.max(axis=0)


def bbox_iou_3d(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_volume = float(np.prod(inter))
    a_volume = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    b_volume = float(np.prod(np.maximum(b_max - b_min, 0.0)))
    union = a_volume + b_volume - inter_volume
    if union <= 0:
        return 0.0
    return inter_volume / union
