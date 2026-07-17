from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np


@dataclass
class UAVFrame:
    """RGB-D observation and pose from one UAV-ON step."""

    rgb: np.ndarray
    depth: np.ndarray
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    step: int
    metadata: Dict[str, Any]

    @classmethod
    def from_observation(cls, observation: Dict[str, Any], camera_index: int = 0) -> "UAVFrame":
        rgb_images = observation.get("rgb") or []
        depth_images = observation.get("depth") or []
        if not rgb_images or not depth_images:
            raise ValueError("Observation does not contain RGB-D images")

        state = observation["sensors"]["state"]
        return cls(
            rgb=_decode_rgb(rgb_images[camera_index]),
            depth=_decode_depth(depth_images[camera_index]),
            position=np.asarray(state["position"], dtype=np.float32),
            quaternion_xyzw=np.asarray(state["quaternionr"], dtype=np.float32),
            step=int(observation.get("step", 0)),
            metadata={
                "description": observation.get("description"),
                "object_name": observation.get("object_name"),
                "object_size": observation.get("object_size"),
            },
        )


def _decode_rgb(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image[..., :3].astype(np.uint8)
    if isinstance(image, (bytes, bytearray)):
        encoded = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Unable to decode RGB image bytes")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    raise TypeError(f"Unsupported RGB image type: {type(image)!r}")


def _decode_depth(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        depth = image.astype(np.float32)
    elif isinstance(image, (bytes, bytearray)):
        encoded = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Unable to decode depth image bytes")
        depth = decoded.astype(np.float32)
    elif isinstance(image, Sequence):
        depth = np.asarray(image, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported depth image type: {type(image)!r}")

    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.max(initial=0) > 255:
        return depth / 1000.0
    return depth / 255.0 * 100.0


def frames_from_episode(episode: Iterable[Dict[str, Any]], camera_index: int = 0) -> List[UAVFrame]:
    frames = []
    for observation in episode:
        if "rgb" in observation and "depth" in observation:
            frames.append(UAVFrame.from_observation(observation, camera_index=camera_index))
    return frames