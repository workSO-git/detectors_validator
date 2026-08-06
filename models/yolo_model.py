import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from .base_model import BaseModel

class YoloModel(BaseModel):
    def __init__(self, model_path, task='seg'):
        self.model = YOLO(model_path)
        self.task = task

    def predict(self, image_path_or_array, conf_threshold=0.25):
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        else:
            img = image_path_or_array
            
        if img is None:
            return {'masks': [], 'boxes': [], 'classes': []}
            
        h, w = img.shape[:2]
        
        results = self.model.predict(img, conf=conf_threshold, verbose=False, retina_masks=True)
        r = results[0]
        
        pred_masks = []
        pred_boxes = []
        pred_classes = []
        
        if self.task == 'seg' and r.masks is not None:
            for i, mask_tensor in enumerate(r.masks.data):
                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_masks.append(mask)
                pred_classes.append(int(r.boxes.cls[i].item()))
        elif self.task == 'det' and r.boxes is not None:
            for i, box_tensor in enumerate(r.boxes.xyxy):
                pred_boxes.append(box_tensor.cpu().numpy().tolist())
                pred_classes.append(int(r.boxes.cls[i].item()))
                
        return {
            'masks': pred_masks,
            'boxes': pred_boxes,
            'classes': pred_classes
        }

    def predict_and_save(self, image_path, save_dir):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        results = self.model.predict(
            source=str(image_path), 
            save=False, 
            show=False
        )
        
        if save_dir:
            for res in results:
                plotted_img = res.plot()
                file_name = Path(res.path).name
                save_path = Path(save_dir) / file_name
                cv2.imwrite(str(save_path), plotted_img)
                print(f"✅ Results saved to {save_path}")
        else:
            print("✅ Inference completed.")
