"""GroundingDINO open-vocabulary detector for UAV-ON scenes.

This wrapper keeps GroundingDINO usage inside the UAV_ON framework while
loading the local source checkout from ../external/GroundingDINO.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert, nms


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GROUNDINGDINO_ROOT = _REPO_ROOT / "external" / "GroundingDINO"
if str(_GROUNDINGDINO_ROOT) not in sys.path:
    sys.path.insert(0, str(_GROUNDINGDINO_ROOT))

try:
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
except Exception:  # pragma: no cover - handled at runtime
    GroundingDinoForObjectDetection = None
    GroundingDinoProcessor = None

from groundingdino.util.inference import load_model, predict  # noqa: E402


DEFAULT_UAVON_CLASSES: List[str] = [
    # UAV-ON target-like categories
    "bus stop",
    "bus shelter",
    "traffic light",
    "traffic signal",
    "bench",
    "park bench",
    "fountain",
    "water fountain",
    "playground",
    "playground equipment",
    "lamp post",
    "street lamp",
    "street sign",
    "stop sign",
    # navigation context
    "road",
    "street",
    "sidewalk",
    "crosswalk",
    "building",
    "house",
    "fence",
    "wall",
    "tree",
    "vegetation",
    "grass",
    "car",
    "truck",
    "vehicle",
    "person",
]


@dataclass
class Detection2D:
    label: str
    confidence: float
    bbox_xyxy: List[float]
    phrase: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class GroundingDINODetector:
    """Small GroundingDINO wrapper returning UAV_ON-friendly detections."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        backend: str = "local",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.config_path = str(config_path or (_GROUNDINGDINO_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
        default_ckpt = _REPO_ROOT / "UAV_ON" / "checkpoints" / "groundingdino_swint_ogc.pth"
        self.checkpoint_path = str(checkpoint_path or default_ckpt)
        self.classes = list(classes or DEFAULT_UAVON_CLASSES)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(f"Loading GroundingDINO on {self.device} with backend={self.backend}...")
        if self.backend == "transformers":
            if GroundingDinoProcessor is None or GroundingDinoForObjectDetection is None:
                raise ImportError(
                    "transformers GroundingDINO backend is unavailable. "
                    "Use backend='local' (the default) with the bundled checkpoint, "
                    "or upgrade transformers to a version that provides GroundingDinoForObjectDetection."
                )
            model_id = "IDEA-Research/grounding-dino-tiny"
            self.processor = GroundingDinoProcessor.from_pretrained(model_id)
            self.model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(self.device)
            self.model.eval()
        elif self.backend == "local":
            if not Path(self.config_path).exists():
                raise FileNotFoundError(f"GroundingDINO config not found: {self.config_path}")
            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(
                    f"GroundingDINO checkpoint not found: {self.checkpoint_path}. "
                    "Download it to UAV_ON/checkpoints/groundingdino_swint_ogc.pth."
                )
            self.model = load_model(self.config_path, self.checkpoint_path, device=self.device)
            self._disable_inference_checkpointing()
        else:
            raise ValueError(f"Unknown GroundingDINO backend: {self.backend!r}; use 'local' or 'transformers'.")
        print(f"✅ GroundingDINO ready with {len(self.classes)} UAV-ON classes")

    def _disable_inference_checkpointing(self) -> None:
        """Disable training-only gradient checkpointing in the local model.

        The supplied GroundingDINO config enables it by default, which emits
        PyTorch warnings on every no-grad inference frame and adds needless
        overhead.  Set the flag recursively after the checkpoint is loaded so
        the vendor config remains untouched.
        """
        disabled = 0
        for module in self.model.modules():
            for attribute in ("use_checkpoint", "use_transformer_ckpt"):
                if hasattr(module, attribute) and getattr(module, attribute):
                    setattr(module, attribute, False)
                    disabled += 1
        if disabled:
            print(f"Disabled gradient checkpointing in {disabled} GroundingDINO modules for inference.")

        # transformers 4.x emits this known compatibility warning from the
        # bundled BERT implementation on every frame. It is not actionable for
        # inference and obscures navigation diagnostics.
        warnings.filterwarnings(
            "ignore",
            message=r"The `device` argument is deprecated and will be removed in v5 of Transformers.*",
            category=FutureWarning,
        )

    @staticmethod
    def caption_from_classes(classes: Iterable[str], target: Optional[str] = None, target_only: bool = False) -> str:
        ordered: List[str] = []
        if target:
            target_norm = target.replace("_", " ").strip().lower()
            ordered.extend([target_norm])
            # Add common aliases for UAV-ON class names.
            if "busstop" in target.lower().replace(" ", "") or target_norm == "bus stop":
                ordered.extend(["bus stop", "bus shelter", "bus station"])
            if "trafficlight" in target.lower().replace(" ", "") or target_norm == "traffic light":
                ordered.extend(["traffic light", "traffic signal", "stoplight"])
            if target_norm == "caravan":
                ordered.extend(["camper trailer", "travel trailer", "caravan trailer", "mobile home trailer"])
            ordered.extend({
                "soccer ball": ["soccer ball", "football", "sports ball"],
                "teapot": ["teapot", "tea kettle", "kettle"],
                "table": ["table", "picnic table", "outdoor table"],
                "chair": ["chair", "outdoor chair", "seat"],
                "rock": ["rock", "boulder", "large stone"],
                "traffic cone": ["traffic cone", "road cone", "safety cone"],
                "stop sign": ["stop sign", "road sign"],
            }.get(target_norm, []))
        if target_only and ordered:
            deduped = []
            for item in ordered:
                if item not in deduped:
                    deduped.append(item)
            return " . ".join(deduped) + " ."
        for c in classes:
            if c not in ordered:
                ordered.append(c)
        return " . ".join(ordered) + " ."

    def detect(
        self,
        image_rgb: np.ndarray,
        target: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        target_only: bool = False,
    ) -> List[Detection2D]:
        """Run GroundingDINO on an RGB image and return xyxy detections."""
        caption = self.caption_from_classes(self.classes, target=target, target_only=target_only)
        if self.backend == "transformers":
            return self._detect_transformers(
                image_rgb=image_rgb,
                caption=caption,
                box_threshold=box_threshold if box_threshold is not None else self.box_threshold,
                text_threshold=text_threshold if text_threshold is not None else self.text_threshold,
                top_k=top_k,
            )

        image_tensor = self._preprocess_image(image_rgb).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold if box_threshold is not None else self.box_threshold,
            text_threshold=text_threshold if text_threshold is not None else self.text_threshold,
            device=self.device,
            remove_combined=True,
        )
        if len(boxes) == 0:
            return []

        h, w = image_rgb.shape[:2]
        boxes_xyxy = box_convert(boxes * torch.tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy")
        detections: List[Detection2D] = []
        for xyxy, score, phrase in zip(boxes_xyxy.tolist(), logits.tolist(), phrases):
            x1, y1, x2, y2 = xyxy
            x1 = float(max(0, min(w - 1, x1)))
            x2 = float(max(0, min(w - 1, x2)))
            y1 = float(max(0, min(h - 1, y1)))
            y2 = float(max(0, min(h - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            label = self._normalize_phrase(phrase)
            detections.append(Detection2D(label=label, confidence=float(score), bbox_xyxy=[x1, y1, x2, y2], phrase=phrase))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        if top_k is not None:
            detections = detections[:top_k]
        return detections

    def detect_tiled(
        self,
        image_rgb: np.ndarray,
        target: str,
        rows: int = 2,
        cols: int = 2,
        overlap: float = 0.20,
        top_k_per_tile: int = 3,
        top_k: int = 8,
        box_threshold: float = 0.20,
        text_threshold: float = 0.14,
    ) -> List[Detection2D]:
        """Detect a small target on overlapping crops and map boxes to the full image."""
        height, width = image_rgb.shape[:2]
        tile_h = int(np.ceil(height / rows))
        tile_w = int(np.ceil(width / cols))
        pad_y = int(tile_h * overlap)
        pad_x = int(tile_w * overlap)
        detections: List[Detection2D] = []
        for row in range(rows):
            for col in range(cols):
                y1 = max(0, row * tile_h - pad_y)
                y2 = min(height, (row + 1) * tile_h + pad_y)
                x1 = max(0, col * tile_w - pad_x)
                x2 = min(width, (col + 1) * tile_w + pad_x)
                crop_detections = self.detect(
                    image_rgb[y1:y2, x1:x2],
                    target=target,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    top_k=top_k_per_tile,
                    target_only=True,
                )
                for detection in crop_detections:
                    bx1, by1, bx2, by2 = detection.bbox_xyxy
                    detections.append(Detection2D(
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox_xyxy=[bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                        phrase=detection.phrase,
                    ))
        if not detections:
            return []
        boxes = torch.tensor([item.bbox_xyxy for item in detections], dtype=torch.float32)
        scores = torch.tensor([item.confidence for item in detections], dtype=torch.float32)
        keep = nms(boxes, scores, 0.45).tolist()
        selected = [detections[index] for index in keep]
        selected.sort(key=lambda item: item.confidence, reverse=True)
        return selected[:top_k]

    def _detect_transformers(
        self,
        image_rgb: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        top_k: Optional[int],
    ) -> List[Detection2D]:
        image_pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        inputs = self.processor(images=image_pil, text=caption, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        detections: List[Detection2D] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            phrase = str(label)
            label_norm = self._normalize_phrase(phrase)
            detections.append(
                Detection2D(
                    label=label_norm,
                    confidence=float(score),
                    bbox_xyxy=[float(v) for v in box.tolist()],
                    phrase=phrase,
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        if top_k is not None:
            detections = detections[:top_k]
        return detections

    @staticmethod
    def _normalize_phrase(phrase: str) -> str:
        phrase = phrase.replace(".", " ").strip().lower()
        if "bus" in phrase and ("stop" in phrase or "shelter" in phrase or "station" in phrase):
            return "bus stop"
        if "traffic" in phrase and ("light" in phrase or "signal" in phrase):
            return "traffic light"
        if "lamp" in phrase or "street lamp" in phrase:
            return "lamp post"
        if "fountain" in phrase:
            return "fountain"
        if "bench" in phrase:
            return "bench"
        if "soccer ball" in phrase or "football" in phrase or "sports ball" in phrase:
            return "soccer ball"
        if "teapot" in phrase or "kettle" in phrase:
            return "teapot"
        if "picnic table" in phrase or "outdoor table" in phrase:
            return "table"
        if "outdoor chair" in phrase or "seat" in phrase:
            return "chair"
        return phrase or "object"

    @staticmethod
    def _preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
        import groundingdino.datasets.transforms as T

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        image_tensor, _ = transform(image_pil, None)
        return image_tensor


def draw_detections(image_rgb: np.ndarray, detections: Sequence[Detection2D], output_path: str) -> None:
    """Save a simple RGB image with detection boxes and labels."""
    image = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{det.label} {det.confidence:.2f}"
        text_w = max(80, len(text) * 8)
        tx1 = x1
        tx2 = min(width - 1, x1 + text_w)
        ty1 = max(0, y1 - 18)
        ty2 = min(height - 1, max(ty1 + 1, y1))
        draw.rectangle([tx1, ty1, tx2, ty2], fill="red")
        draw.text((tx1 + 2, ty1 + 1), text, fill="white", font=font)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
