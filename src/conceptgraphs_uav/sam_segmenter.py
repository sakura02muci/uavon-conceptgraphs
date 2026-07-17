"""SAM bbox-prompted instance segmentation for UAV-ON observations."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


class SAMSegmenter:
    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM on {self.device}: {model_id}...")
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("✅ SAM ready")

    def segment_bbox(self, image_rgb: np.ndarray, bbox_xyxy: Sequence[float]) -> np.ndarray | None:
        """Return the highest-IoU boolean mask for one bbox prompt."""
        image = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        box = [[float(value) for value in bbox_xyxy]]
        inputs = self.processor(image, input_boxes=[box], return_tensors="pt")
        original_sizes = inputs["original_sizes"]
        reshaped_sizes = inputs["reshaped_input_sizes"]
        model_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes.cpu(), reshaped_sizes.cpu()
        )[0]
        scores = outputs.iou_scores[0, 0].detach().cpu()
        if masks.ndim == 4:
            masks = masks[0]
        best = int(torch.argmax(scores).item())
        mask = masks[best].numpy().astype(bool)
        if mask.ndim == 3:
            mask = mask[0]
        return mask if int(mask.sum()) >= 16 else None
