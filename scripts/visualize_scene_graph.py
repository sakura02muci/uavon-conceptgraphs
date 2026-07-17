"""Visualize ConceptGraph results on UAV-ON data."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize_scene_graph(graph_path: str, rgbd_dir: str, output_dir: str):
    """Create comprehensive visualization of scene graph and RGB-D data."""
    
    # Load scene graph
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # === 1. 3D Scene Graph Visualization ===
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: 3D positions
    ax1 = fig.add_subplot(221, projection='3d')
    positions = np.array([node['centroid'] for node in graph['nodes']])
    
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                c='blue', marker='o', s=100, alpha=0.6)
    
    # Draw edges
    for edge in graph['edges']:
        src_idx = int(edge['source'].split('_')[-1])
        tgt_idx = int(edge['target'].split('_')[-1])
        if src_idx < len(positions) and tgt_idx < len(positions):
            ax1.plot([positions[src_idx, 0], positions[tgt_idx, 0]],
                    [positions[src_idx, 1], positions[tgt_idx, 1]],
                    [positions[src_idx, 2], positions[tgt_idx, 2]],
                    'r-', alpha=0.3, linewidth=1)
    
    # Add labels
    for i, node in enumerate(graph['nodes']):
        ax1.text(positions[i, 0], positions[i, 1], positions[i, 2],
                f"N{i}", fontsize=8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Scene Graph ({len(graph["nodes"])} nodes, {len(graph["edges"])} edges)')
    ax1.grid(True)
    
    # Plot 2: Top-down view
    ax2 = fig.add_subplot(222)
    ax2.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, alpha=0.6)
    
    for edge in graph['edges']:
        src_idx = int(edge['source'].split('_')[-1])
        tgt_idx = int(edge['target'].split('_')[-1])
        if src_idx < len(positions) and tgt_idx < len(positions):
            ax2.plot([positions[src_idx, 0], positions[tgt_idx, 0]],
                    [positions[src_idx, 1], positions[tgt_idx, 1]],
                    'r-', alpha=0.3, linewidth=1)
    
    for i in range(len(positions)):
        ax2.text(positions[i, 0], positions[i, 1], f"N{i}", fontsize=8)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View')
    ax2.grid(True)
    ax2.axis('equal')
    
    # Plot 3: Statistics
    ax3 = fig.add_subplot(223)
    ax3.axis('off')
    
    # Compute statistics
    if len(graph['edges']) > 0:
        distances = [edge['distance'] for edge in graph['edges']]
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
    else:
        avg_dist = min_dist = max_dist = 0
    
    stats_text = f"""
    Scene Graph Statistics
    ━━━━━━━━━━━━━━━━━━━━━━
    Nodes: {len(graph['nodes'])}
    Edges: {len(graph['edges'])}
    
    Distance Statistics:
    - Average: {avg_dist:.2f}m
    - Min: {min_dist:.2f}m
    - Max: {max_dist:.2f}m
    
    Node Labels:
    """
    
    label_counts = {}
    for node in graph['nodes']:
        label = node['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in label_counts.items():
        stats_text += f"\n  {label}: {count}"
    
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    # Plot 4: Distance distribution
    ax4 = fig.add_subplot(224)
    if len(graph['edges']) > 0:
        ax4.hist(distances, bins=10, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Count')
        ax4.set_title('Edge Distance Distribution')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No edges', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'scene_graph_3d.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved 3D visualization to {output_path / 'scene_graph_3d.png'}")
    plt.close()
    
    # === 2. RGB-D Grid Visualization ===
    rgbd_path = Path(rgbd_dir)
    rgb_files = sorted(rgbd_path.glob('rgb_*.png'))
    depth_files = sorted(rgbd_path.glob('depth_*.npy'))
    
    if rgb_files and depth_files:
        # Show up to 10 frames for large datasets, 6 for small
        total_frames = min(len(rgb_files), len(depth_files))
        max_frames = 10 if total_frames > 10 else 6
        n_frames = min(total_frames, max_frames)
        
        fig, axes = plt.subplots(2, n_frames, figsize=(3*n_frames, 6))
        if n_frames == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_frames):
            # RGB
            rgb_img = cv2.imread(str(rgb_files[i]))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(rgb_img)
            axes[0, i].set_title(f'Frame {i} RGB')
            axes[0, i].axis('off')
            
            # Depth
            depth_img = np.load(depth_files[i])
            depth_vis = axes[1, i].imshow(depth_img, cmap='viridis')
            axes[1, i].set_title(f'Frame {i} Depth')
            axes[1, i].axis('off')
            plt.colorbar(depth_vis, ax=axes[1, i], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_path / 'rgbd_grid.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved RGB-D grid to {output_path / 'rgbd_grid.png'}")
        plt.close()
    
    # === 3. Create detailed node info ===
    with open(output_path / 'node_details.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ConceptGraph Node Details\n")
        f.write("=" * 70 + "\n\n")
        
        for i, node in enumerate(graph['nodes']):
            f.write(f"Node {i}: {node['node_id']}\n")
            f.write(f"  Label: {node['label']}\n")
            f.write(f"  Centroid: [{node['centroid'][0]:.2f}, {node['centroid'][1]:.2f}, {node['centroid'][2]:.2f}]\n")
            f.write(f"  BBox: [{node['bbox_min'][0]:.2f}, {node['bbox_min'][1]:.2f}, {node['bbox_min'][2]:.2f}] -> ")
            f.write(f"[{node['bbox_max'][0]:.2f}, {node['bbox_max'][1]:.2f}, {node['bbox_max'][2]:.2f}]\n")
            f.write(f"  Observations: {node['observations']}\n")
            f.write(f"  Confidence: {node['confidence']}\n")
            if node.get('caption'):
                f.write(f"  Caption: {node['caption']}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Edge Details\n")
        f.write("=" * 70 + "\n\n")
        
        for edge in graph['edges']:
            f.write(f"{edge['source']} --[{edge['relation']}, {edge['distance']:.2f}m]--> {edge['target']}\n")
    
    print(f"✅ Saved node details to {output_path / 'node_details.txt'}")
    
    print(f"\n{'='*70}")
    print(f"Visualization complete! Files saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ConceptGraph results")
    parser.add_argument("--graph", type=str, default="./barnyard_scene_graph.json", 
                       help="Path to scene graph JSON")
    parser.add_argument("--rgbd_dir", type=str, default="./airsim_collected",
                       help="Path to RGB-D data directory")
    parser.add_argument("--output_dir", type=str, default="./visualization",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    visualize_scene_graph(
        graph_path=args.graph,
        rgbd_dir=args.rgbd_dir,
        output_dir=args.output_dir
    )
