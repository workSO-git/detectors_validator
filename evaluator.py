import cv2
import numpy as np
import yaml
from pathlib import Path
from metrics import compute_box_iou, compute_mask_iou, calculate_precision_recall

def polygon_to_mask(polygon, width, height):
    """Convert normalized YOLO polygon [x1, y1, x2, y2...] to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) == 0:
        return mask
    pts = []
    for i in range(0, len(polygon), 2):
        x = int(polygon[i] * width)
        y = int(polygon[i+1] * height)
        pts.append([x, y])
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 1)
    return mask

def box_to_absolute(box, width, height):
    """Convert normalized YOLO box [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x_center, y_center = cx * width, cy * height
    box_w, box_h = w * width, h * height
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)
    return [x1, y1, x2, y2]

class GenericEvaluator:
    def __init__(self, model_adapter, task='seg'):
        self.model = model_adapter
        self.task = task
        
    def evaluate_dataset(self, data_yaml_path, iou_threshold=0.5, conf_threshold=0.25):
        """
        Generic evaluation loop calculating mIoU, Precision, and Recall.
        """
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_info = yaml.safe_load(f)
            
        base_dir = Path(data_yaml_path).parent
        val_img_dir = base_dir / data_info.get('val', 'images/val')
        val_lbl_dir = val_img_dir.parent.parent / 'labels' / val_img_dir.name
        
        if not val_img_dir.exists():
            print(f"Error: Validation image directory not found: {val_img_dir}")
            return
            
        images = list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png'))
        print(f"Found {len(images)} images for validation.")
        
        ious = []
        tps, fps, fns = 0, 0, 0
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            
            # Load Ground Truth (currently supports YOLO txt format)
            lbl_path = val_lbl_dir / (img_path.stem + '.txt')
            gt_masks = []
            gt_boxes = []
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = list(map(float, line.strip().split()))
                        if self.task == 'seg' and len(parts) > 5:
                            gt_masks.append(polygon_to_mask(parts[1:], w, h))
                        elif self.task == 'det' and len(parts) == 5:
                            gt_boxes.append(box_to_absolute(parts[1:], w, h))
            
            # Run Prediction using generic model adapter
            preds = self.model.predict(img, conf_threshold=conf_threshold)
            pred_masks = preds.get('masks', [])
            pred_boxes = preds.get('boxes', [])

            # Match and calculate IoU
            matched_gt = set()
            
            if self.task == 'seg':
                for p_mask in pred_masks:
                    best_iou, best_gt_idx = 0, -1
                    for i, g_mask in enumerate(gt_masks):
                        if i in matched_gt: continue
                        iou = compute_mask_iou(p_mask, g_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    if best_iou >= iou_threshold:
                        tps += 1
                        matched_gt.add(best_gt_idx)
                        ious.append(best_iou)
                    else:
                        fps += 1
                fns += len(gt_masks) - len(matched_gt)
                
            elif self.task == 'det':
                for p_box in pred_boxes:
                    best_iou, best_gt_idx = 0, -1
                    for i, g_box in enumerate(gt_boxes):
                        if i in matched_gt: continue
                        iou = compute_box_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    if best_iou >= iou_threshold:
                        tps += 1
                        matched_gt.add(best_gt_idx)
                        ious.append(best_iou)
                    else:
                        fps += 1
                fns += len(gt_boxes) - len(matched_gt)

        mean_iou = np.mean(ious) if ious else 0.0
        precision, recall = calculate_precision_recall(tps, fps, fns)
        
        print("\n" + "="*40)
        print(f"📊 Evaluation Results (Threshold: IoU >= {iou_threshold})")
        print("="*40)
        print(f"  Mean IoU:  {mean_iou:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print("="*40)
        
        return {'mIoU': mean_iou, 'precision': precision, 'recall': recall}

    def evaluate_single(self, source_path, save_dir=None):
        """
        Run inference on a single image and visualize via adapter.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            print(f"Error: Source file not found: {source_path}")
            return
            
        print(f"🚀 Running inference on {source_path.name}...")
        self.model.predict_and_save(source_path, save_dir)
