"""
Aggregate UAV-ON evaluation results across all scenes.

Generates a comprehensive summary report with:
- Overall metrics (SR, SPL, DtG)
- Per-scene breakdown
- Comparison with UAV-ON baselines
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_scene_results(results_dir: Path, strategy: str) -> Dict:
    """Load all scene results."""
    scene_dir = results_dir / f"uavon_{strategy}"
    
    if not scene_dir.exists():
        print(f"❌ Results directory not found: {scene_dir}")
        return {}
    
    all_results = {}
    result_files = list(scene_dir.glob("*.json"))
    
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        scene_name = result_file.stem
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                all_results[scene_name] = data
                print(f"  ✓ {scene_name}: {data['summary']['num_episodes']} episodes")
        except Exception as e:
            print(f"  ✗ {scene_name}: Failed to load ({e})")
    
    return all_results


def aggregate_metrics(all_results: Dict) -> Dict:
    """Compute overall metrics."""
    all_episodes = []
    
    for scene_name, scene_data in all_results.items():
        episodes = scene_data.get('episodes', [])
        for ep in episodes:
            ep['scene'] = scene_name
            all_episodes.append(ep)
    
    if not all_episodes:
        return {
            'total_episodes': 0,
            'success_rate': 0.0,
            'mean_spl': 0.0,
            'mean_distance_to_goal': 0.0,
            'mean_path_length': 0.0
        }
    
    # Compute metrics
    successes = [ep['success'] for ep in all_episodes]
    spls = [ep['spl'] for ep in all_episodes]
    distances = [ep['min_distance_to_goal'] for ep in all_episodes]
    path_lengths = [ep['path_length'] for ep in all_episodes]
    
    return {
        'total_episodes': len(all_episodes),
        'num_scenes': len(all_results),
        'success_rate': np.mean(successes),
        'mean_spl': np.mean(spls),
        'std_spl': np.std(spls),
        'mean_distance_to_goal': np.mean(distances),
        'std_distance_to_goal': np.std(distances),
        'mean_path_length': np.mean(path_lengths),
        'std_path_length': np.std(path_lengths),
        'per_scene': {
            scene: {
                'success_rate': np.mean([ep['success'] for ep in data['episodes']]),
                'mean_spl': np.mean([ep['spl'] for ep in data['episodes']]),
                'num_episodes': len(data['episodes'])
            }
            for scene, data in all_results.items()
        }
    }


def format_report(metrics: Dict, strategy: str) -> str:
    """Generate formatted report."""
    report = []
    report.append("="*70)
    report.append(f"UAV-ON Evaluation Report: {strategy.upper()} Strategy")
    report.append("="*70)
    report.append("")
    
    # Overall metrics
    report.append("## Overall Metrics")
    report.append(f"- Total episodes: {metrics['total_episodes']}")
    report.append(f"- Number of scenes: {metrics['num_scenes']}")
    report.append(f"- Success Rate (SR): {metrics['success_rate']:.1%}")
    report.append(f"- Mean SPL: {metrics['mean_spl']:.3f} ± {metrics['std_spl']:.3f}")
    report.append(f"- Mean Distance to Goal: {metrics['mean_distance_to_goal']:.2f}m ± {metrics['std_distance_to_goal']:.2f}m")
    report.append(f"- Mean Path Length: {metrics['mean_path_length']:.1f}m ± {metrics['std_path_length']:.1f}m")
    report.append("")
    
    # Per-scene breakdown
    report.append("## Per-Scene Results")
    report.append("")
    report.append(f"{'Scene':<20} {'Episodes':>10} {'SR':>10} {'SPL':>10}")
    report.append("-" * 52)
    
    for scene, scene_metrics in sorted(metrics['per_scene'].items()):
        sr = scene_metrics['success_rate']
        spl = scene_metrics['mean_spl']
        n_ep = scene_metrics['num_episodes']
        report.append(f"{scene:<20} {n_ep:>10} {sr:>9.1%} {spl:>10.3f}")
    
    report.append("")
    
    # Baseline comparison
    report.append("## Comparison with UAV-ON Baselines")
    report.append("")
    report.append(f"{'Method':<20} {'SR':>10} {'SPL':>10}")
    report.append("-" * 42)
    report.append(f"{'Random':<20} {'~5%':>10} {'~0.02':>10}")
    report.append(f"{'FMM':<20} {'~15%':>10} {'~0.08':>10}")
    report.append(f"{'CLIP-H':<20} {'~35%':>10} {'~0.22':>10}")
    report.append(f"{f'ConceptGraphs ({strategy})':<20} {metrics['success_rate']:>9.1%} {metrics['mean_spl']:>10.3f}")
    report.append("")
    
    # Analysis
    report.append("## Analysis")
    if metrics['success_rate'] > 0.35:
        report.append("✅ Performance exceeds CLIP-H baseline!")
    elif metrics['success_rate'] > 0.15:
        report.append("⚠️  Performance between FMM and CLIP-H")
    else:
        report.append("❌ Performance below FMM baseline")
    
    report.append("")
    report.append(f"Mean SPL of {metrics['mean_spl']:.3f} indicates path efficiency.")
    
    if metrics['mean_spl'] > 0.22:
        report.append("Path planning is reasonably efficient.")
    else:
        report.append("Path planning could be improved (many detours).")
    
    report.append("")
    report.append("="*70)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Aggregate UAV-ON results")
    parser.add_argument("--strategy", type=str, default="clip",
                       help="Strategy name (clip/deepseek/baseline)")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="Results directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output report file (default: print to stdout)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load results
    print("Loading results...")
    all_results = load_scene_results(results_dir, args.strategy)
    
    if not all_results:
        print("❌ No results found!")
        return
    
    # Aggregate metrics
    print("\nComputing metrics...")
    metrics = aggregate_metrics(all_results)
    
    # Generate report
    report = format_report(metrics, args.strategy)
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\n✅ Report saved to: {output_path}")
    else:
        print("\n")
        print(report)
    
    # Save JSON summary
    json_output = results_dir / f"uavon_{args.strategy}" / "summary.json"
    with open(json_output, 'w') as f:
        json.dump({
            'strategy': args.strategy,
            'metrics': metrics
        }, f, indent=2)
    print(f"✅ JSON summary saved to: {json_output}")


if __name__ == "__main__":
    main()
