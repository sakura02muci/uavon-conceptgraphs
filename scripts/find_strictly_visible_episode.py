"""Find UAV-ON episodes whose target is genuinely rendered in the initial view.

This uses AirSim instance segmentation as the visibility oracle.  A target is
accepted only when its mesh contributes enough pixels to the initial camera
image; merely projecting the navigation goal inside the camera FOV is not
considered visible.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import airsim
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def response_rgb(response) -> np.ndarray:
    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    return image.reshape(response.height, response.width, 3)


def target_mask_from_segmentation(segmentation: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the non-background instance mask after all other IDs are set to 0."""
    colors, counts = np.unique(segmentation.reshape(-1, 3), axis=0, return_counts=True)
    background = colors[int(np.argmax(counts))]
    mask = np.any(segmentation != background[None, None, :], axis=2)
    return mask, [int(value) for value in background]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-target-pixels", type=int, default=80)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--only-target", default=None,
                        help="Optional semantic target name, e.g. Caravan")
    parser.add_argument("--search-controlled-start", action="store_true",
                        help="Search a circle around the target for a truly visible start pose")
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    args = parser.parse_args()

    episodes = json.loads(Path(args.dataset).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port)
    client.confirmConnection()
    client.enableApiControl(True)

    # A binary instance image: background ID 0, current target ID 1.
    client.simSetSegmentationObjectID(".*", 0, True)
    time.sleep(0.5)  # Unreal applies stencil-ID updates on the render thread.
    records = []
    scan_episodes = episodes[args.start_index:args.end_index]
    for episode in scan_episodes:
        target_name = str(episode["true_name"]).strip()
        if args.only_target and target_name.lower() != args.only_target.lower():
            continue
        object_name = str(episode["object_name"])
        matched = bool(client.simSetSegmentationObjectID(object_name, 1, False))
        time.sleep(0.2)
        start = episode["start_pose"]["start_position"]
        quat = episode["start_pose"]["start_quaternionr"]
        pose = airsim.Pose(
            airsim.Vector3r(*start),
            airsim.Quaternionr(*quat),
        )
        client.simSetVehiclePose(pose, True)
        time.sleep(0.25)
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
        ])
        rgb = response_rgb(responses[0])
        segmentation = response_rgb(responses[1])
        depth = np.asarray(responses[2].image_data_float, dtype=np.float32).reshape(
            responses[2].height, responses[2].width
        )
        mask, background_color = target_mask_from_segmentation(segmentation)
        ys, xs = np.where(mask)
        pixels = int(mask.sum()) if matched else 0
        visible = matched and pixels >= args.min_target_pixels
        bbox = None
        target_depth = None
        if pixels:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            valid_depth = depth[mask & np.isfinite(depth) & (depth > 0)]
            if valid_depth.size:
                target_depth = {
                    "median_m": float(np.median(valid_depth)),
                    "min_m": float(np.min(valid_depth)),
                    "max_m": float(np.max(valid_depth)),
                }

        record = {
            "episode_id": str(episode["episode_id"]),
            "target_name": target_name,
            "object_name": object_name,
            "segmentation_object_matched": matched,
            "visible": visible,
            "target_pixels": pixels,
            "target_fraction": float(pixels / mask.size),
            "bbox_xyxy": bbox,
            "target_depth": target_depth,
            "segmentation_background_rgb": background_color,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
        (output_dir / "visibility_report.partial.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False)
        )

        if visible:
            evidence = rgb.copy()
            overlay = evidence.copy()
            overlay[mask] = (255, 40, 40)
            evidence = cv2.addWeighted(evidence, 0.65, overlay, 0.35, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(evidence, f"{object_name}: {pixels} px", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / f"episode_{episode['episode_id']}_visibility.png"),
                        cv2.cvtColor(evidence, cv2.COLOR_RGB2BGR))

        # Restore this target to background before examining the next instance.
        client.simSetSegmentationObjectID(object_name, 0, False)

    (output_dir / "visibility_report.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )
    visible_records = [record for record in records if record["visible"]]
    print(f"Strictly visible: {len(visible_records)}/{len(records)}")

    if args.search_controlled_start and not visible_records:
        candidates = [ep for ep in episodes if not args.only_target or
                      str(ep["true_name"]).strip().lower() == args.only_target.lower()]
        if not candidates:
            raise RuntimeError("No matching target episode is available as a template")
        template = candidates[0]
        object_name = str(template["object_name"])
        target_pose = client.simGetObjectPose(object_name).position
        target = np.array([target_pose.x_val, target_pose.y_val, target_pose.z_val])
        client.simSetSegmentationObjectID(object_name, 1, False)
        best = None
        for radius in (12.0, 18.0, 25.0):
            for altitude in (-4.0, -7.0, -10.0):
                for bearing_deg in range(0, 360, 15):
                    bearing = np.deg2rad(bearing_deg)
                    position = target + np.array([
                        radius * np.cos(bearing), radius * np.sin(bearing), altitude - target[2]
                    ])
                    yaw = np.arctan2(target[1] - position[1], target[0] - position[0])
                    orientation = airsim.to_quaternion(0.0, 0.0, float(yaw))
                    pose = airsim.Pose(airsim.Vector3r(*position), orientation)
                    client.simSetVehiclePose(pose, True)
                    time.sleep(0.06)
                    responses = client.simGetImages([
                        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
                        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
                    ])
                    rgb = response_rgb(responses[0])
                    segmentation = response_rgb(responses[1])
                    mask, _ = target_mask_from_segmentation(segmentation)
                    pixels = int(mask.sum())
                    if best is None or pixels > best["pixels"]:
                        best = {"pixels": pixels, "pose": pose, "position": position,
                                "orientation": orientation, "rgb": rgb, "mask": mask,
                                "depth_response": responses[2], "radius": radius,
                                "bearing_deg": bearing_deg}

        if best is None or best["pixels"] < args.min_target_pixels:
            raise RuntimeError(f"No strictly visible controlled pose found; best={best and best['pixels']} px")
        mask = best["mask"]
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        depth = np.asarray(best["depth_response"].image_data_float, dtype=np.float32).reshape(
            best["depth_response"].height, best["depth_response"].width
        )
        target_depth = depth[mask & np.isfinite(depth) & (depth > 0)]
        evidence = best["rgb"].copy()
        overlay = evidence.copy()
        overlay[mask] = (255, 40, 40)
        evidence = cv2.addWeighted(evidence, 0.65, overlay, 0.35, 0)
        cv2.rectangle(evidence, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / "controlled_start_visibility.png"),
                    cv2.cvtColor(evidence, cv2.COLOR_RGB2BGR))

        controlled = json.loads(json.dumps(template))
        controlled["episode_id"] = f"{template['episode_id']}_strict_visible"
        controlled["start_pose"]["start_position"] = [float(v) for v in best["position"]]
        controlled["start_pose"]["start_quaternionr"] = [
            float(best["orientation"].x_val), float(best["orientation"].y_val),
            float(best["orientation"].z_val), float(best["orientation"].w_val),
        ]
        straight_distance = float(np.linalg.norm(best["position"] - target))
        controlled["info"]["euclidean_distance"] = straight_distance
        controlled["info"]["geodesic_distance"] = straight_distance
        controlled["visibility_oracle"] = {
            "method": "AirSim instance segmentation",
            "target_pixels": int(best["pixels"]),
            "bbox_xyxy": bbox,
            "target_depth_median": float(np.median(target_depth)),
            "radius_xy_m": float(best["radius"]),
            "bearing_deg": int(best["bearing_deg"]),
        }
        (output_dir / "controlled_visible_episode.json").write_text(
            json.dumps([controlled], indent=2, ensure_ascii=False)
        )
        print("Controlled episode:", json.dumps(controlled["visibility_oracle"], ensure_ascii=False))


if __name__ == "__main__":
    main()
