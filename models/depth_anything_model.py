import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from .base_model import BaseModel


class DepthAnythingModel(BaseModel):
    """
    Depth estimation model using Depth-Anything-V2-Small-hf (HuggingFace).
    
    Converts depth map into a foreground binary mask by thresholding:
    pixels with normalized depth >= depth_threshold are considered "near" (foreground).
    
    Returns:
        - masks: [binary_mask] — single mask of the near/foreground region
        - boxes: [] — not used
        - classes: [0] — single class
    """
    
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    
    def __init__(self, model_path=None, task='seg', depth_threshold=0.5, invert=False):
        """
        Args:
            model_path: Ignored (model is auto-downloaded from HuggingFace).
                        Can optionally be a local path to a cached model dir.
            task: Always 'seg' for this model (produces a mask).
            depth_threshold: Normalized [0..1] threshold.
                             Pixels with relative depth >= threshold are treated as foreground.
                             Default 0.5 means "closer half of the depth range".
            invert: If True, select FAR objects instead of NEAR objects.
        """
        self.task = task
        self.depth_threshold = depth_threshold
        self.invert = invert
        self._pipeline = None  # lazy load
        self._model_path = model_path  # may be a local cache dir
        
    def _load(self):
        if self._pipeline is not None:
            return
            
        from transformers import pipeline as hf_pipeline
        
        model_id = self._model_path if (self._model_path and Path(self._model_path).is_dir()) else self.MODEL_ID
        
        device = 0 if torch.cuda.is_available() else -1  # GPU if available
        
        print(f"[DepthAnything] Loading model from '{model_id}' (device={'cuda' if device == 0 else 'cpu'})...")
        self._pipeline = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
        )
        print("[DepthAnything] Model loaded.")
        
    def _depth_to_mask(self, depth_array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Convert a float32 depth array to a binary mask based on threshold."""
        # Normalize to [0, 1]
        d_min, d_max = depth_array.min(), depth_array.max()
        if d_max - d_min < 1e-6:
            return np.zeros((target_h, target_w), dtype=np.uint8)
        
        depth_norm = (depth_array - d_min) / (d_max - d_min)
        
        # In typical depth maps: HIGHER value = FARTHER away.
        # We want the NEAR objects, so threshold on LOW depth values.
        # If invert=False → near → depth_norm <= (1 - depth_threshold)
        # If invert=True  → far  → depth_norm >= depth_threshold
        if not self.invert:
            mask = (depth_norm <= (1.0 - self.depth_threshold)).astype(np.uint8)
        else:
            mask = (depth_norm >= self.depth_threshold).astype(np.uint8)
            
        # Resize to original image size
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def predict(self, image_path_or_array, conf_threshold=0.25):
        self._load()
        
        # Load image
        if isinstance(image_path_or_array, (str, Path)):
            pil_img = Image.open(str(image_path_or_array)).convert("RGB")
            h, w = pil_img.height, pil_img.width
        else:
            # numpy array (BGR from cv2)
            arr = image_path_or_array
            h, w = arr.shape[:2]
            pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        
        # Run depth estimation
        result = self._pipeline(pil_img)
        
        # result["depth"] is a PIL Image in mode "I" (32-bit int) or "F" (float)
        depth_pil = result["depth"]
        depth_array = np.array(depth_pil, dtype=np.float32)
        
        # Build binary mask
        mask = self._depth_to_mask(depth_array, h, w)
        
        return {
            'masks': [mask],
            'boxes': [],
            'classes': [0],
        }
    
    def predict_and_save(self, image_path, save_dir):
        """Run inference and save a side-by-side visualization: original | depth | mask."""
        self._load()
        
        image_path = Path(image_path)
        if save_dir:
            from pathlib import Path as P
            P(save_dir).mkdir(parents=True, exist_ok=True)
            
        # Load image
        pil_img = Image.open(str(image_path)).convert("RGB")
        orig_bgr = cv2.imread(str(image_path))
        h, w = orig_bgr.shape[:2]
        
        # Depth estimation
        result = self._pipeline(pil_img)
        depth_pil = result["depth"]
        depth_array = np.array(depth_pil, dtype=np.float32)
        
        # Normalize depth for visualization (0..255 grayscale)
        d_min, d_max = depth_array.min(), depth_array.max()
        if d_max > d_min:
            depth_vis = ((depth_array - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros((h, w), dtype=np.uint8)
        depth_vis_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_vis_bgr = cv2.resize(depth_vis_bgr, (w, h))
        
        # Binary mask overlay
        mask = self._depth_to_mask(depth_array, h, w)
        overlay = orig_bgr.copy()
        overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([0, 120, 255]) * 0.5).astype(np.uint8)
        
        # Concatenate horizontally: original | depth colormap | mask overlay
        vis = np.concatenate([orig_bgr, depth_vis_bgr, overlay], axis=1)
        
        if save_dir:
            save_path = Path(save_dir) / image_path.name
            cv2.imwrite(str(save_path), vis)
            print(f"Saved: {save_path}")
        else:
            print("Inference completed (no save_dir specified).")
