"""
Build scene graph from UAV-ON episode with GOAL-DIRECTED navigation.

This script navigates toward the goal and builds a scene graph from the trajectory.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import airsim
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R_scipy

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from conceptgraphs_uav import UAVFrame
from conceptgraphs_uav.graph import ConceptGraphBuilder
from conceptgraphs_uav.clip_detector import CLIPDetector
from conceptgraphs_uav.io import save_scene_graph


def navigate_and_build_graph(client, episode, max_steps=100, save_dir=None):
    """Navigate toward goal while building scene graph."""
    
    episode_id = episode['episode_id']
    target_name = episode['true_name'].strip()
    map_name = episode['map_name']
    goal_pose = np.array(episode['pose'][0])
    
    print(f"\n{'='*70}")
    print(f"Goal-Directed Navigation with Scene Graph Building")
    print(f"{'='*70}")
    print(f"Episode ID: {episode_id}")
    print(f"Target: {target_name}")
    print(f"Goal: {goal_pose[:2].round(1)}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*70}\n")
    
    # Initialize
    print("Initializing CLIP detector...")
    clip_detector = CLIPDetector()
    
    print("Initializing ConceptGraphs builder...")
    graph_builder = ConceptGraphBuilder(
        merge_iou=0.15,
        fov_degrees=90.0
    )
    
    # Reset to start pose
    start_pos = episode['start_pose']['start_position']
    start_quat = episode['start_pose']['start_quaternionr']
    
    client.enableApiControl(True)
    client.armDisarm(True)
    
    pose = airsim.Pose(
        airsim.Vector3r(start_pos[0], start_pos[1], start_pos[2]),
        airsim.Quaternionr(start_quat[0], start_quat[1], start_quat[2], start_quat[3])
    )
    client.simSetVehiclePose(pose, True)
    time.sleep(0.5)
    
    client.takeoffAsync().join()
    time.sleep(0.5)
    
    # Navigation variables
    frames_collected = 0
    min_distance = float('inf')
    trajectory = []
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    for step in range(max_steps):
        # Get observation
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        
        if len(responses) < 2:
            print(f"Step {step}: Failed to get images")
            break
        
        # Extract RGB and depth
        rgb_response = responses[0]
        rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)
        
        depth_response = responses[1]
        depth = np.array(depth_response.image_data_float, dtype=np.float32)
        depth = depth.reshape(depth_response.height, depth_response.width)
        
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
        
        # Calculate distance and direction to goal
        direction_to_goal = goal_pose - position
        distance_to_goal = np.linalg.norm(direction_to_goal)
        min_distance = min(min_distance, distance_to_goal)
        
        # CLIP detection
        predictions = clip_detector.classify_image(rgb, top_k=1)
        label, confidence = predictions[0]
        
        if step % 5 == 0:
            print(f"Step {step:3d}: Position={position[:2].round(1)}, "
                  f"Dist={distance_to_goal:.1f}m, Detected='{label}' ({confidence:.3f})")
        
        # Create UAVFrame
        frame = UAVFrame(
            rgb=rgb,
            depth=depth,
            position=position,
            quaternion_xyzw=orientation,
            step=step,
            metadata={'clip_label': label, 'clip_confidence': float(confidence)}
        )
        
        # Add to graph builder
        graph_builder.add_frame(frame, label=label)
        frames_collected += 1
        trajectory.append(position.copy())
        
        # Save frame (every 5 steps)
        if save_dir and step % 5 == 0:
            Image.fromarray(rgb).save(save_path / f"rgb_{step:04d}.png")
            np.save(save_path / f"depth_{step:04d}.npy", depth)
            
            metadata = {
                'step': step,
                'position': position.tolist(),
                'orientation': orientation.tolist(),
                'distance_to_goal': float(distance_to_goal),
                'clip_label': label,
                'clip_confidence': float(confidence)
            }
            with open(save_path / f"meta_{step:04d}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Check if reached goal
        if distance_to_goal < 5.0:
            print(f"\n🎯 Reached goal at step {step}!")
            break
        
        # GOAL-DIRECTED NAVIGATION
        # Calculate current yaw
        r = R_scipy.from_quat(orientation)
        yaw = r.as_euler('xyz')[2]
        
        # Calculate angle to goal
        goal_angle = np.arctan2(direction_to_goal[1], direction_to_goal[0])
        angle_diff = goal_angle - yaw
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        angle_diff_deg = np.degrees(angle_diff)
        
        # Decision: turn toward goal or move forward
        if abs(angle_diff_deg) > 30:
            # Need to turn
            turn_rate = 30 if angle_diff_deg > 0 else -30
            client.rotateByYawRateAsync(turn_rate, 1).join()
            if step % 10 == 0:
                print(f"         🔄 Turning {angle_diff_deg:.0f}° toward goal")
        else:
            # Move forward toward goal
            velocity = 3 if distance_to_goal > 20 else 2
            client.moveByVelocityAsync(velocity, 0, 0, 1, 
                                      drivetrain=airsim.DrivetrainType.ForwardOnly).join()
        
        time.sleep(0.1)
    
    # Finalize scene graph
    print(f"\n{'='*70}")
    print(f"Finalizing scene graph from {frames_collected} frames...")
    print(f"{'='*70}")
    
    scene_graph = graph_builder.finalize()
    
    # Statistics
    num_nodes = len(scene_graph.graph.nodes)
    num_edges = len(scene_graph.graph.edges)
    
    trajectory_length = 0
    for i in range(1, len(trajectory)):
        trajectory_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
    
    print(f"\n✅ Scene Graph Built Successfully!")
    print(f"   Frames collected: {frames_collected}")
    print(f"   Nodes: {num_nodes}")
    print(f"   Edges: {num_edges}")
    print(f"   Min distance to goal: {min_distance:.1f}m")
    print(f"   Trajectory length: {trajectory_length:.1f}m")
    
    # Land
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    return scene_graph, frames_collected, min_distance, trajectory_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="UAV-ON dataset JSON")
    parser.add_argument("--episode-id", type=int, default=0, help="Episode ID")
    parser.add_argument("--max-steps", type=int, default=50, help="Max navigation steps")
    parser.add_argument("--output", default="scene_graph_goal_directed.json", help="Output file")
    parser.add_argument("--save-frames", default=None, help="Save frames to directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r') as f:
        episodes = json.load(f)
    
    if args.episode_id >= len(episodes):
        print(f"❌ Episode {args.episode_id} not found (max: {len(episodes)-1})")
        return
    
    episode = episodes[args.episode_id]
    print(f"✅ Found episode {args.episode_id}")
    print(f"   Target: {episode['true_name']}")
    print(f"   Map: {episode['map_name']}")
    
    # Connect to AirSim
    print("\nConnecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected!")
    
    # Navigate and build graph
    scene_graph, num_frames, min_dist, traj_length = navigate_and_build_graph(
        client, episode, args.max_steps, args.save_frames
    )
    
    # Save scene graph
    print(f"\nSaving scene graph to {args.output}...")
    save_scene_graph(scene_graph, args.output)
    print("✅ Saved!")
    
    # Visualize
    if args.visualize:
        vis_dir = Path(args.output).stem + "_visualizations"
        print(f"\nGenerating visualizations...")
        # Simple visualization without external function
        Path(vis_dir).mkdir(exist_ok=True)
        
        # Save node details
        with open(Path(vis_dir) / "node_details.txt", 'w') as f:
            f.write("="*70 + "\n")
            f.write("ConceptGraph Node Details (Goal-Directed)\n")
            f.write("="*70 + "\n\n")
            for node_id, data in scene_graph.graph.nodes(data=True):
                f.write(f"Node {node_id}:\n")
                f.write(f"  Label: {data.get('label', 'unknown')}\n")
                centroid = data.get('centroid', [0, 0, 0])
                f.write(f"  Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]\n")
                f.write(f"  Observations: {data.get('count', 0)}\n\n")
        
        print(f"✅ Visualizations saved to {vis_dir}/")
    
    # Summary
    num_nodes = len(scene_graph.graph.nodes)
    num_edges = len(scene_graph.graph.edges)
    
    print(f"\n{'='*70}")
    print(f"Scene Graph Summary (Goal-Directed Navigation)")
    print(f"{'='*70}")
    print(f"Episode: {args.episode_id} ({episode['map_name']})")
    print(f"Target: {episode['true_name']}")
    print(f"Frames collected: {num_frames}")
    print(f"Scene graph nodes: {num_nodes}")
    print(f"Scene graph edges: {num_edges}")
    print(f"Min distance to goal: {min_dist:.1f}m")
    print(f"Trajectory length: {traj_length:.1f}m")
    print(f"Output: {args.output}")
    if args.visualize:
        print(f"Visualizations: {vis_dir}/")
    print(f"{'='*70}")
    
    # Node summary
    from collections import Counter
    labels = [data.get('label', 'unknown') for _, data in scene_graph.graph.nodes(data=True)]
    label_counts = Counter(labels)
    
    print(f"\nNode Summary:")
    print(f"{'='*70}")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
