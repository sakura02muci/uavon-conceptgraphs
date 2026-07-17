"""Collect RGB-D and pose data from UAV-ON environment without running full evaluation."""
import json
import os
import sys
from pathlib import Path

import numpy as np

# Set simulator port before importing
os.environ.setdefault('SIMULATOR_PORT', '30001')

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.env_uav import AirVLNENV


def collect_data(dataset_path: str, output_dir: str, max_episodes: int = 5, port: int = 30001):
    """Collect RGB-D observations and poses from environment initialization."""
    
    # Update args simulator_tool_port
    from src.common.param import args
    args.simulator_tool_port = port
    
    env = AirVLNENV(
        batch_size=1,
        dataset_path=dataset_path,
        save_path=None,
        seed=42
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    collected = 0
    episode_idx = 0
    
    print(f"Loaded {len(env.data)} episodes from {dataset_path}")
    print(f"Will collect up to {max_episodes} episodes")
    
    while collected < max_episodes and episode_idx < len(env.data):
        batch = env.next_minibatch(skip_scenes=[])
        if batch is None:
            break
            
        # Get initial observation
        obs_list = env.get_obs()
        if not obs_list or len(obs_list) == 0:
            episode_idx += env.batch_size
            continue
            
        observations, done, collision, oracle_success = obs_list[0]
        
        # Extract data from latest observation
        latest_obs = observations[-1]
        if 'rgb' not in latest_obs or 'depth' not in latest_obs:
            episode_idx += env.batch_size
            continue
            
        # Save episode data
        episode_data = {
            'episode_id': batch[0].get('task_id', f'episode_{episode_idx}'),
            'map_name': batch[0].get('map_name', 'unknown'),
            'object_name': latest_obs.get('object_name'),
            'description': latest_obs.get('description'),
            'position': latest_obs['sensors']['state']['position'],
            'quaternionr': latest_obs['sensors']['state']['quaternionr'],
            'rgb_count': len(latest_obs['rgb']),
            'depth_count': len(latest_obs['depth']),
        }
        
        episode_dir = output_path / f"episode_{collected:04d}"
        episode_dir.mkdir(exist_ok=True)
        
        # Save RGB images
        rgb_dir = episode_dir / "rgb"
        rgb_dir.mkdir(exist_ok=True)
        for i, rgb_img in enumerate(latest_obs['rgb']):
            if isinstance(rgb_img, bytes):
                (rgb_dir / f"{i:03d}.jpg").write_bytes(rgb_img)
            else:
                np.save(rgb_dir / f"{i:03d}.npy", np.asarray(rgb_img))
        
        # Save depth images
        depth_dir = episode_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        for i, depth_img in enumerate(latest_obs['depth']):
            if isinstance(depth_img, bytes):
                (depth_dir / f"{i:03d}.jpg").write_bytes(depth_img)
            else:
                np.save(depth_dir / f"{i:03d}.npy", np.asarray(depth_img))
        
        # Save metadata
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(episode_data, f, indent=2)
        
        print(f"Collected episode {collected}: {episode_data['map_name']} - {episode_data['object_name']}")
        
        collected += 1
        episode_idx += env.batch_size
    
    print(f"\nCollected {collected} episodes to {output_path}")
    env.delete_VectorEnvUtil()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect RGB-D data from UAV-ON without full evaluation", conflict_handler='resolve')
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--output_dir", type=str, default="./collected_rgbd", help="Output directory")
    parser.add_argument("--max_episodes", type=int, default=5, help="Maximum episodes to collect")
    parser.add_argument("--port", type=int, default=30001, help="Simulator port")
    
    args_collect, _ = parser.parse_known_args()
    
    collect_data(
        dataset_path=args_collect.dataset_path,
        output_dir=args_collect.output_dir,
        max_episodes=args_collect.max_episodes,
        port=args_collect.port
    )
