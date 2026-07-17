"""CLIP-based pseudo detector for UAV-ON scenes."""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))


class CLIPDetector:
    """CLIP-based image classifier for scene understanding."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading CLIP model on {device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Common objects in UAV outdoor scenes + UAV-ON specific targets
        self.categories = [
            # Buildings
            "building", "house", "skyscraper",
            # Vehicles
            "car", "vehicle", "truck", "pickup truck", "SUV", "limousine", "rusty car",
            # Roads and paths
            "road", "street", "pavement",
            # Nature
            "tree", "vegetation", "grass",
            # Sky
            "sky", "cloud",
            # People
            "person", "pedestrian",
            # Traffic infrastructure
            "traffic light", "traffic signal", "stoplight",
            "street sign", "stop sign",
            # Structures
            "fence", "wall",
            # Areas
            "parking lot", "sidewalk",
            # UAV-ON specific targets
            "bus stop", "bus shelter", "bus station",
            "bench", "park bench",
            "fountain", "water fountain",
            "playground", "playground equipment",
            "lamp post", "street lamp",
            "picnic table", "table", "cooking stove", "stove", "grill", "oven",
            "sign", "signboard", "trash can", "mailbox", "statue"
        ]
        
        # Encode text categories
        text_inputs = torch.cat([clip.tokenize(self._prompt_for_category(c)) for c in self.categories]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        print(f"✅ CLIP ready with {len(self.categories)} categories")

    @staticmethod
    def _prompt_for_category(category: str) -> str:
        article = "an" if category[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"a drone view photo of {article} {category} in an outdoor navigation scene"
    
    def classify_image(self, image: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Classify entire image and return top-k predictions."""
        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
        
        # Preprocess and encode
        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features @ self.text_features.T).squeeze(0)
            probs = similarity.softmax(dim=0)
        
        # Get top-k
        top_probs, top_indices = probs.topk(top_k)
        results = [(self.categories[idx], prob.item()) for idx, prob in zip(top_indices, top_probs)]
        
        return results

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Return a normalized CLIP feature for one object crop.

        Object-map association needs a stable visual descriptor rather than the
        softmax category score used by :meth:`classify_image`.
        """
        image_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model.encode_image(image_input)
            feature /= feature.norm(dim=-1, keepdim=True)
        return feature.squeeze(0).detach().cpu().numpy().astype(np.float32)
    
    def detect_grid_regions(self, image: np.ndarray, grid_size: Tuple[int, int] = (2, 2)) -> Dict:
        """
        Divide image into grid and classify each region.
        This is a simple pseudo-detection without actual masks.
        """
        h, w = image.shape[:2]
        grid_h, grid_w = grid_size
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        
        detections = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < grid_h - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < grid_w - 1 else w
                
                # Crop region
                region = image[y1:y2, x1:x2]
                
                # Classify
                predictions = self.classify_image(region, top_k=1)
                if predictions:
                    label, confidence = predictions[0]
                    
                    detections.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],  # xyxy format
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    })
        
        return {
            'detections': detections,
            'image_shape': image.shape[:2]
        }
    
    def get_scene_summary(self, image: np.ndarray) -> str:
        """Get natural language summary of the scene."""
        # Whole image classification
        top_predictions = self.classify_image(image, top_k=3)
        
        summary = "Scene contains: "
        items = [f"{label} ({conf:.2f})" for label, conf in top_predictions]
        summary += ", ".join(items)
        
        return summary


def test_clip_detector():
    """Test CLIP detector on sample data."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--grid_size", type=int, nargs=2, default=[2, 2])
    args = parser.parse_args()
    
    # Load image
    image = np.array(Image.open(args.image))
    
    # Create detector
    detector = CLIPDetector()
    
    # Whole image classification
    print("\n" + "=" * 70)
    print("Whole Image Classification")
    print("=" * 70)
    predictions = detector.classify_image(image, top_k=5)
    for i, (label, conf) in enumerate(predictions, 1):
        print(f"{i}. {label:30s} {conf:.4f}")
    
    # Grid detection
    print("\n" + "=" * 70)
    print(f"Grid Detection ({args.grid_size[0]}x{args.grid_size[1]})")
    print("=" * 70)
    detections = detector.detect_grid_regions(image, tuple(args.grid_size))
    for i, det in enumerate(detections['detections'], 1):
        print(f"{i}. {det['label']:20s} @ [{det['bbox'][0]:3d},{det['bbox'][1]:3d},{det['bbox'][2]:3d},{det['bbox'][3]:3d}] conf={det['confidence']:.3f}")
    
    # Scene summary
    print("\n" + "=" * 70)
    print("Scene Summary")
    print("=" * 70)
    print(detector.get_scene_summary(image))
    print()


if __name__ == "__main__":
    test_clip_detector()
