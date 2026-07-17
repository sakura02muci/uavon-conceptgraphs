"""DeepSeek integration for UAV navigation decision making."""
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_deepseek_key


class DeepSeekPlanner:
    """Use DeepSeek for navigation planning based on scene graph."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")
        
        # Try to get API key from: 1) parameter, 2) .env file, 3) environment variable
        self.api_key = api_key or get_deepseek_key() or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Please:\n"
                "1. Copy .env.template to .env and fill in your key, OR\n"
                "2. Set DEEPSEEK_API_KEY environment variable, OR\n"
                "3. Pass api_key parameter"
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        
        print(f"✅ DeepSeek planner initialized (model: {model})")
    
    def create_scene_summary(self, scene_graph: Dict, current_position: List[float]) -> str:
        """Convert scene graph to natural language summary."""
        nodes = scene_graph.get('nodes', [])
        
        summary = f"Current UAV position: [{current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}]\n\n"
        summary += f"Visible objects in scene ({len(nodes)} nodes):\n"
        
        for i, node in enumerate(nodes, 1):
            label = node['label']
            centroid = node['centroid']
            caption = node.get('caption', '')
            
            # Calculate relative position
            rel_pos = [centroid[0] - current_position[0],
                      centroid[1] - current_position[1],
                      centroid[2] - current_position[2]]
            dist = (rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)**0.5
            
            summary += f"{i}. {label} at [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}] "
            summary += f"(~{dist:.1f}m away"
            
            # Direction
            if rel_pos[0] > 0:
                summary += ", ahead"
            else:
                summary += ", behind"
            
            if abs(rel_pos[1]) > 1:
                summary += ", left" if rel_pos[1] < 0 else ", right"
            
            summary += ")"
            
            if caption:
                summary += f" - {caption}"
            
            summary += "\n"
        
        return summary
    
    def decide_action(
        self,
        scene_graph: Dict,
        current_position: List[float],
        target_object: str,
        history: List[str] = None
    ) -> Dict:
        """Use DeepSeek to decide next navigation action."""
        
        scene_summary = self.create_scene_summary(scene_graph, current_position)
        
        # Build prompt
        system_prompt = """You are an expert UAV navigation planner. Your task is to help a drone navigate to find a target object using scene information.

Available actions:
- forward: move 2m forward
- backward: move 2m backward
- rotl: rotate 30° left
- rotr: rotate 30° right
- ascend: move 1m up
- descend: move 1m down
- stop: stop navigation (when target found or task failed)

Respond with one valid JSON object only, no markdown, no extra text:
{
  "action": "forward|backward|rotl|rotr|ascend|descend|stop",
  "reasoning": "brief explanation of why this action"
}"""
        
        history_text = ""
        if history:
            history_text = "\n\nPrevious actions: " + ", ".join(history[-5:])
        
        user_prompt = f"""Goal: Find and navigate to '{target_object}'

{scene_summary}{history_text}

What should the drone do next?"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=120,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            elif not content.startswith("{"):
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if match:
                    content = match.group(0)
            
            result = json.loads(content)
            
            return {
                'action': result.get('action', 'forward'),
                'reasoning': result.get('reasoning', ''),
                'raw_response': content
            }
            
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            # Fallback to baseline
            return {
                'action': 'forward',
                'reasoning': f'API error, using fallback: {str(e)}',
                'raw_response': ''
            }

    def decide_subgoal(
        self,
        scene_graph: Dict,
        current_position: List[float],
        target_object: str,
        target_description: str = "",
        history: List[str] = None,
    ) -> Dict:
        """Choose a semantic graph node or exploration direction, not a motor action."""
        nodes = scene_graph.get("nodes", [])
        compact_nodes = [
            {
                "node_id": node.get("node_id"),
                "label": node.get("label"),
                "caption": node.get("caption", ""),
                "confidence": node.get("confidence", 0.0),
                "observations": node.get("observations", 1),
                "distance": node.get("distance", 0.0),
            }
            for node in nodes
        ]
        system_prompt = """You choose high-level semantic subgoals for UAV object navigation.
Never output motor commands. Select one observed node to inspect, or select an exploration direction.
Return one JSON object only:
{"subgoal_type":"object_node|explore", "node_id":"node id or null", "direction":"forward|left|right", "reasoning":"brief reason"}"""
        user_prompt = (
            f"Target name: {target_object}\nTarget description: {target_description}\n"
            f"Current position: {current_position}\nGraph edges: {scene_graph.get('edges', [])[:60]}\n"
            f"Candidate nodes: {compact_nodes}\nRecent subgoals: {(history or [])[-5:]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=160,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            node_ids = {str(node.get("node_id")) for node in nodes}
            node_id = str(result.get("node_id")) if result.get("node_id") is not None else None
            if result.get("subgoal_type") == "object_node" and node_id not in node_ids:
                node_id = None
            return {
                "subgoal_type": "object_node" if node_id else "explore",
                "node_id": node_id,
                "direction": result.get("direction", "forward"),
                "reasoning": result.get("reasoning", ""),
                "raw_response": content,
            }
        except Exception as exc:
            return {
                "subgoal_type": "explore",
                "node_id": None,
                "direction": "forward",
                "reasoning": f"planner fallback: {exc}",
                "raw_response": "",
            }


def test_deepseek_planner():
    """Test DeepSeek planner with sample scene graph."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_graph", type=str, default="./scene_graph_clip.json")
    parser.add_argument("--target", type=str, default="vehicle")
    parser.add_argument("--api_key", type=str, help="DeepSeek API key")
    args = parser.parse_args()
    
    # Load scene graph
    with open(args.scene_graph, 'r') as f:
        scene_graph = json.load(f)
    
    # Create planner
    planner = DeepSeekPlanner(api_key=args.api_key)
    
    # Test decision
    current_pos = [10.0, 1.0, -1.0]
    
    print("\n" + "=" * 70)
    print("DeepSeek Navigation Test")
    print("=" * 70)
    print(f"Target: {args.target}")
    print(f"Current position: {current_pos}")
    print()
    
    decision = planner.decide_action(
        scene_graph=scene_graph,
        current_position=current_pos,
        target_object=args.target,
        history=['forward', 'forward', 'rotl']
    )
    
    print("Decision:")
    print(f"  Action: {decision['action']}")
    print(f"  Reasoning: {decision['reasoning']}")
    print()
    print("Raw response:")
    print(decision['raw_response'])
    print()


if __name__ == "__main__":
    test_deepseek_planner()
