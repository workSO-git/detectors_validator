import cv2
import numpy as np
import torch
from pathlib import Path
from .base_model import BaseModel

class SmpModel(BaseModel):
    """
    Adapter for models trained with segmentation_models_pytorch (SMP).
    Expects a standard UNet with a ResNet backbone by default.
    """
    TARGET_SIZE = (512, 512)  # Must match training script

    def __init__(self, model_path, task='seg', encoder_name='resnet34', classes=1):
        import segmentation_models_pytorch as smp
        
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_path_obj = Path(model_path)
        self.is_onnx = False
        self.is_ts = False
        
        if model_path_obj.suffix == '.onnx':
            print(f"[SmpModel] Loading ONNX model from {model_path}...")
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == 'cuda' else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(str(model_path), providers=providers)
            self.is_onnx = True
        elif model_path_obj.suffix == '.pt':
            print(f"[SmpModel] Loading TorchScript model from {model_path}...")
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
            self.is_ts = True
        else:
            print(f"[SmpModel] Initializing UNet({encoder_name}) on {self.device}...")
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=classes,
            )
            print(f"[SmpModel] Loading weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
        # NOTE: NO ImageNet normalization — model was trained with simple /255 only
        print("[SmpModel] Model ready.")

    def predict(self, image_path_or_array, conf_threshold=0.5):
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        else:
            img = image_path_or_array

        if img is None:
            return {'masks': [], 'boxes': [], 'classes': []}

        h, w = img.shape[:2]

        # Preprocessing EXACTLY as in training script (4_inference.py):
        # BGR -> RGB, resize to 512x512, divide by 255 — NO ImageNet normalization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.TARGET_SIZE)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_normalized, (2, 0, 1))           # HWC -> CHW
        img_tensor = np.expand_dims(img_tensor, axis=0)                 # -> BCHW
        img_tensor = torch.tensor(img_tensor).to(self.device)

        with torch.no_grad():
            if self.is_onnx:
                ort_inputs = {self.sess.get_inputs()[0].name: img_tensor.cpu().numpy()}
                output = self.sess.run(None, ort_inputs)[0]
                output = torch.tensor(output).to(self.device)
            else:
                output = self.model(img_tensor)
            probs = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()

        # Threshold at 0.5 (same as training script)
        mask_512 = (probs >= conf_threshold).astype(np.uint8)

        # Resize back to original image dimensions
        mask = cv2.resize(mask_512, (w, h), interpolation=cv2.INTER_NEAREST)

        return {
            'masks': [mask],
            'boxes': [],
            'classes': [0]
        }

    def predict_and_save(self, image_path, save_dir):
        if save_dir:
            from pathlib import Path as P
            P(save_dir).mkdir(parents=True, exist_ok=True)
            
        image_path = Path(image_path)
        orig_bgr = cv2.imread(str(image_path))
        
        result = self.predict(orig_bgr)
        mask = result['masks'][0] if len(result['masks']) > 0 else np.zeros(orig_bgr.shape[:2], dtype=np.uint8)
        
        overlay = orig_bgr.copy()
        # Draw mask as semi-transparent green
        overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        
        vis = np.concatenate([orig_bgr, overlay], axis=1)
        
        if save_dir:
            save_path = Path(save_dir) / image_path.name
            cv2.imwrite(str(save_path), vis)
            print(f"✅ Saved to {save_path}")
        else:
            print("✅ Inference completed.")
