"""
Build ConceptGraphs scene graph from UAV-ON episode navigation.

This script runs a UAV-ON episode, collects RGB-D frames during navigation,
and builds a ConceptGraphs scene graph with CLIP semantic labels.
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

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from conceptgraphs_uav import ConceptGraphBuilder, UAVFrame
from conceptgraphs_uav.clip_detector import CLIPDetector
from conceptgraphs_uav.io import save_scene_graph


def collect_and_build_graph(client, episode, max_steps=100, save_dir=None):
    """Navigate episode and build scene graph from collected frames."""
    
    episode_id = episode['episode_id']
    target_name = episode['true_name'].strip()
    map_name = episode['map_name']
    
    print(f"\n{'='*70}")
    print(f"Building Scene Graph from UAV-ON Episode")
    print(f"{'='*70}")
    print(f"Episode ID: {episode_id}")
    print(f"Map: {map_name}")
    print(f"Target: {target_name}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*70}\n")
    
    # Initialize components
    print("Initializing CLIP detector...")
    clip_detector = CLIPDetector()
    
    print("Initializing ConceptGraphs builder...")
    graph_builder = ConceptGraphBuilder(
        fov_degrees=90.0,
        point_stride=8,
        max_depth=80.0,
        merge_iou=0.15
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
    
    # Collect frames and build graph
    frames_collected = 0
    
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
        
        # CLIP detection
        predictions = clip_detector.classify_image(rgb, top_k=1)
        label, confidence = predictions[0]
        
        if step % 5 == 0:
            print(f"Step {step:3d}: Position={position[:2].round(1)}, "
                  f"Detected='{label}' (conf={confidence:.3f})")
        
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
        
        # Optionally save frame
        if save_dir and step % 5 == 0:
            Image.fromarray(rgb).save(save_path / f"rgb_{step:04d}.png")
            np.save(save_path / f"depth_{step:04d}.npy", depth)
            
            metadata = {
                'step': step,
                'position': position.tolist(),
                'orientation': orientation.tolist(),
                'clip_label': label,
                'clip_confidence': float(confidence)
            }
            with open(save_path / f"meta_{step:04d}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Simple exploration: move forward, occasionally rotate
        if step % 10 == 9:
            client.rotateByYawRateAsync(30, 1).join()
        else:
            client.moveByVelocityAsync(3, 0, 0, 1, 
                                      drivetrain=airsim.DrivetrainType.ForwardOnly).join()
        
        time.sleep(0.1)
    
    # Finalize scene graph
    print(f"\n{'='*70}")
    print(f"Finalizing scene graph from {frames_collected} frames...")
    print(f"{'='*70}")
    
    scene_graph = graph_builder.finalize()
    
    # Print summary
    num_nodes = len(scene_graph.graph.nodes)
    num_edges = len(scene_graph.graph.edges)
    
    print(f"\n✅ Scene Graph Built Successfully!")
    print(f"   Frames: {frames_collected}")
    print(f"   Nodes: {num_nodes}")
    print(f"   Edges: {num_edges}")
    
    # Land
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    
    return scene_graph, frames_collected


def main():
    parser = argparse.ArgumentParser(description="Build scene graph from UAV-ON episode")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to UAV-ON dataset JSON file")
    parser.add_argument("--episode-id", type=str, default="0",
                       help="Episode ID to use (default: 0)")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Maximum steps to collect")
    parser.add_argument("--output", type=str, default="uavon_scene_graph.json",
                       help="Output scene graph JSON file")
    parser.add_argument("--save-frames", type=str, default=None,
                       help="Directory to save collected frames (optional)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'r') as f:
        episodes = json.load(f)
    
    # Find target episode
    episode = None
    for ep in episodes:
        if ep['episode_id'] == args.episode_id:
            episode = ep
            break
    
    if episode is None:
        print(f"❌ Episode {args.episode_id} not found!")
        return
    
    print(f"✅ Found episode {args.episode_id}")
    print(f"   Target: {episode['true_name']}")
    print(f"   Map: {episode['map_name']}")
    
    # Connect to AirSim
    print("\nConnecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected!")
    
    # Run collection and building
    scene_graph, num_frames = collect_and_build_graph(
        client, episode,
        max_steps=args.max_steps,
        save_dir=args.save_frames
    )
    
    # Save scene graph
    print(f"\nSaving scene graph to {args.output}...")
    save_scene_graph(scene_graph, args.output)
    print(f"✅ Saved!")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        viz_dir = Path(args.output).stem + "_visualizations"
        
        try:
            import subprocess
            result = subprocess.run([
                'python', 'scripts/visualize_scene_graph.py',
                '--graph', args.output,
                '--output_dir', viz_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Visualizations saved to {viz_dir}/")
            else:
                print(f"⚠️  Visualization failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not generate visualizations: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Scene Graph Summary")
    print(f"{'='*70}")
    print(f"Episode: {episode['episode_id']} ({episode['map_name']})")
    print(f"Target: {episode['true_name']}")
    print(f"Frames collected: {num_frames}")
    print(f"Scene graph nodes: {len(scene_graph.graph.nodes)}")
    print(f"Scene graph edges: {len(scene_graph.graph.edges)}")
    print(f"Output: {args.output}")
    if args.save_frames:
        print(f"Frames saved: {args.save_frames}/")
    if args.visualize:
        print(f"Visualizations: {viz_dir}/")
    print(f"{'='*70}")
    
    # Print node summary
    print(f"\nNode Summary:")
    print(f"{'='*70}")
    
    from collections import Counter
    labels = [data['label'] for _, data in scene_graph.graph.nodes(data=True)]
    label_counts = Counter(labels)
    
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
