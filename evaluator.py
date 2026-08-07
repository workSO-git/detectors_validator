import cv2
import numpy as np
import yaml
from pathlib import Path
import time
import json
import urllib.parse
from metrics import (
    compute_box_iou, compute_mask_iou,
    calculate_precision_recall, calculate_f1_score,
    get_mask_centroid, get_box_centroid,
    calculate_jitter, calculate_flicker_rate
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def polygon_to_mask(polygon, width, height):
    """Convert normalized YOLO polygon [x1,y1,x2,y2,...] to a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 6:  # need at least 3 points
        return mask
    pts = np.array(
        [[int(polygon[i] * width), int(polygon[i + 1] * height)]
         for i in range(0, len(polygon), 2)],
        dtype=np.int32
    )
    cv2.fillPoly(mask, [pts], 1)
    return mask


def box_to_absolute(box, width, height):
    """Convert normalized YOLO box [cx,cy,w,h] to absolute [x1,y1,x2,y2]."""
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    x2 = int((cx + bw / 2) * width)
    y2 = int((cy + bh / 2) * height)
    return [x1, y1, x2, y2]


def _load_gt_from_file(lbl_path, task, width, height):
    """Load GT masks or boxes from a YOLO-format .txt label file."""
    gt_masks, gt_boxes = [], []
    if not lbl_path.exists():
        return gt_masks, gt_boxes
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if not parts:
                continue
            if task == 'seg' and len(parts) > 5:
                gt_masks.append(polygon_to_mask(parts[1:], width, height))
            elif task == 'det' and len(parts) == 5:
                gt_boxes.append(box_to_absolute(parts[1:], width, height))
    return gt_masks, gt_boxes


_JSON_CACHE = {}

def _load_gt_from_json(json_path, filename, width, height, task='seg'):
    """Load GT masks or boxes from a custom JSON annotation file.
       Returns (gt_masks, gt_boxes, is_found).
    """
    gt_masks, gt_boxes = [], []
    json_path = Path(json_path)
    if not json_path.exists():
        return gt_masks, gt_boxes, False
        
    path_str = str(json_path)
    if path_str not in _JSON_CACHE:
        with open(json_path, 'r', encoding='utf-8') as f:
            _JSON_CACHE[path_str] = json.load(f)
            
    data = _JSON_CACHE[path_str]
    
    target_stem = Path(filename).stem
    
    import re
    clean_target = re.sub(r'[^a-zA-Z0-9]', '', target_stem).lower()
    
    # Find the specific image by matching the uploaded filename stem with the JSON uuid or link
    image_data = None
    for item in data:
        item_uuid = item.get("uuid", "")
        clean_uuid = re.sub(r'[^a-zA-Z0-9]', '', item_uuid).lower()
        
        link = item.get("link", "")
        unquoted_link = urllib.parse.unquote(urllib.parse.unquote(link))
        clean_link = re.sub(r'[^a-zA-Z0-9]', '', unquoted_link).lower()
        
        # Check if the filename matches uuid, or is a substring of uuid, or uuid is a substring of filename, or it's in the link
        if clean_uuid and (clean_target == clean_uuid or clean_target in clean_uuid or clean_uuid in clean_target):
            image_data = item
            break
        if clean_target and clean_target in clean_link:
            image_data = item
            break
            
    if not image_data:
        return gt_masks, gt_boxes, False

    # The JSON annotations were drawn on 1280x720 resolution images.
    # We must scale them to the actual uploaded image dimensions.
    scale_x = width / 1280.0
    scale_y = height / 720.0
    print(f"DEBUG: Scaled JSON polygons for {filename} by {scale_x}x{scale_y}")

    # Extract polygons (for segmentation)
    if task == 'seg':
        has_seg_annotation = False
        
        # Explicitly marked as having no sky
        if image_data.get("no_sky") is True:
            has_seg_annotation = True
            
        if "ignore_polygons" in image_data:
            has_seg_annotation = True
            for poly_points in image_data["ignore_polygons"]:
                mask = np.zeros((height, width), dtype=np.uint8)
                # JSON format is usually [[[x,y], [x,y], ...]] or [[x,y], [x,y]]
                # Depending on nesting, extract the list of points
                if len(poly_points) > 0:
                    pts = np.array(poly_points, dtype=np.float32)
                    # Ensure shape is (N, 2)
                    if pts.ndim == 3 and pts.shape[0] == 1:
                        pts = pts[0]
                    if len(pts) >= 3:
                        # Scale coordinates
                        pts[:, 0] *= scale_x
                        pts[:, 1] *= scale_y
                        pts = pts.astype(np.int32)
                        
                        cv2.fillPoly(mask, [pts], 1)
                        gt_masks.append(mask)
                        
        if not has_seg_annotation:
            return gt_masks, gt_boxes, False

    # Extract rects (for detection) - stored as flat list [x, y, w, h, x, y, w, h...] in ignorerects
    if task == 'det':
        if "ignorerects" not in image_data:
            return gt_masks, gt_boxes, False
            
        rects_flat = image_data["ignorerects"]
        if isinstance(rects_flat, list):
            for i in range(0, len(rects_flat), 4):
                if i + 3 < len(rects_flat):
                    rx, ry, rw, rh = rects_flat[i:i+4]
                    # Convert to [x1, y1, x2, y2] and scale
                    x1 = int(rx * scale_x)
                    y1 = int(ry * scale_y)
                    x2 = int((rx + rw) * scale_x)
                    y2 = int((ry + rh) * scale_y)
                    gt_boxes.append([x1, y1, x2, y2])
                    
    return gt_masks, gt_boxes, True


def _match_predictions(pred_list, gt_list, iou_fn, iou_threshold):
    """
    Greedy matching of predictions to GT.
    Returns (tp, fp, fn, matched_ious).
    """
    matched_gt = set()
    tp, fp = 0, 0
    matched_ious = []

    for pred in pred_list:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_list):
            if i in matched_gt:
                continue
            iou = iou_fn(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_list) - len(matched_gt)
    return tp, fp, fn, matched_ious


# ─── Evaluator ───────────────────────────────────────────────────────────────

class GenericEvaluator:
    def __init__(self, model_adapter, task='seg'):
        self.model = model_adapter
        self.task = task

    def _iou_fn(self):
        return compute_mask_iou if self.task == 'seg' else compute_box_iou

    # ── Dataset evaluation ───────────────────────────────────────────────────
    def evaluate_dataset(self, data_yaml_path, split='val', iou_threshold=0.5, conf_threshold=0.25):
        """
        Evaluate on a full dataset defined by a YOLO data.yaml.
        Calculates mIoU, Precision, Recall, F1.
        """
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_info = yaml.safe_load(f)
        base_dir = Path(data_yaml_path).parent
        split_path_rel = data_info.get(split, f'images/{split}')
        val_img_dir = base_dir / split_path_rel
        val_lbl_dir = val_img_dir.parent.parent / 'labels' / val_img_dir.name

        if not val_img_dir.exists():
            print(f"Error: Validation image dir not found: {val_img_dir}")
            return None

        images = list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png'))
        print(f"Found {len(images)} images for validation.")

        all_ious, inference_times = [], []
        total_tp, total_fp, total_fn = 0, 0, 0
        iou_fn = self._iou_fn()

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            lbl_path = val_lbl_dir / (img_path.stem + '.txt')
            gt_masks, gt_boxes = _load_gt_from_file(lbl_path, self.task, w, h)

            t0    = time.perf_counter()
            preds = self.model.predict(img, conf_threshold=conf_threshold)
            inference_times.append((time.perf_counter() - t0) * 1000)

            pred_list = preds.get('masks', []) if self.task == 'seg' else preds.get('boxes', [])
            gt_list   = gt_masks if self.task == 'seg' else gt_boxes

            tp, fp, fn, ious = _match_predictions(pred_list, gt_list, iou_fn, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_ious.extend(ious)

        mean_iou  = float(np.mean(all_ious)) if all_ious else 0.0
        precision, recall = calculate_precision_recall(total_tp, total_fp, total_fn)
        f1        = calculate_f1_score(precision, recall)
        avg_ms    = float(np.mean(inference_times)) if inference_times else 0.0

        self._print_dataset_results(mean_iou, precision, recall, f1, avg_ms, iou_threshold)
        return {'mIoU': mean_iou, 'precision': precision, 'recall': recall,
                'f1': f1, 'avg_infer_ms': avg_ms}

    # ── Single image ─────────────────────────────────────────────────────────
    def evaluate_single(self, source_path, save_dir=None):
        """Run inference on a single image/video and save result."""
        source_path = Path(source_path)
        if not source_path.exists():
            print(f"Error: Source not found: {source_path}")
            return
        print(f"🚀 Running inference on {source_path.name}...")
        self.model.predict_and_save(source_path, save_dir)

    # ── Video evaluation ─────────────────────────────────────────────────────
    def evaluate_video(self, video_path, labels_dir, iou_threshold=0.5, conf_threshold=0.25):
        """
        Evaluate model on a video with optional frame-by-frame GT annotations.
        Labels directory should contain 0000.txt, 0001.txt, ... in YOLO format.
        Always reports: Avg FPS, Temporal Jitter, Flicker Rate.
        If labels_dir provided: also mIoU, Precision, Recall, F1.
        """
        video_path = Path(video_path)
        labels_dir = Path(labels_dir) if labels_dir else None

        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🚀 Evaluating video: {video_path.name} ({total_frames} frames)")

        all_ious, inference_times = [], []
        total_tp, total_fp, total_fn = 0, 0, 0
        centroids, detections_presence = [], []
        iou_fn = self._iou_fn()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Load GT for this frame if labels_dir is given
            gt_masks, gt_boxes = [], []
            if labels_dir:
                lbl_path = labels_dir / f"{frame_idx:04d}.txt"
                gt_masks, gt_boxes = _load_gt_from_file(lbl_path, self.task, w, h)

            # Run inference
            t0    = time.perf_counter()
            preds = self.model.predict(frame, conf_threshold=conf_threshold)
            inference_times.append(time.perf_counter() - t0)

            pred_masks = preds.get('masks', [])
            pred_boxes = preds.get('boxes', [])
            pred_list  = pred_masks if self.task == 'seg' else pred_boxes
            gt_list    = gt_masks   if self.task == 'seg' else gt_boxes

            # Detect presence & centroid (for jitter/flicker — always calculated)
            has_detection = len(pred_list) > 0
            detections_presence.append(has_detection)
            if has_detection:
                c = get_mask_centroid(pred_list[0]) if self.task == 'seg' else get_box_centroid(pred_list[0])
                centroids.append(c)
            else:
                centroids.append(None)

            # IoU matching only if GT available
            if labels_dir and gt_list:
                tp, fp, fn, ious = _match_predictions(pred_list, gt_list, iou_fn, iou_threshold)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_ious.extend(ious)
            elif labels_dir:
                # GT file exists but no annotations → any predictions are FP
                total_fp += len(pred_list)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

        cap.release()

        # Aggregate
        total_time = sum(inference_times)
        avg_fps    = frame_idx / total_time if total_time > 0 else 0.0
        jitter     = calculate_jitter(centroids)
        flicker    = calculate_flicker_rate(detections_presence)

        mean_iou, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        if labels_dir and all_ious:
            mean_iou          = float(np.mean(all_ious))
            precision, recall = calculate_precision_recall(total_tp, total_fp, total_fn)
            f1                = calculate_f1_score(precision, recall)

        self._print_video_results(mean_iou, precision, recall, f1,
                                  avg_fps, jitter, flicker, has_gt=bool(labels_dir))
        res = {'avg_fps': avg_fps, 'jitter': jitter, 'flicker': flicker}
        if labels_dir:
            res.update({'mIoU': mean_iou, 'precision': precision,
                        'recall': recall, 'f1': f1})
        return res

    # ── Print helpers ─────────────────────────────────────────────────────────
    def _print_dataset_results(self, miou, precision, recall, f1, avg_ms, thresh):
        sep = '=' * 45
        print(f"\n{sep}")
        print(f"📊 Dataset Evaluation Results (IoU >= {thresh})")
        print(sep)
        print(f"  Mean IoU:         {miou:.4f}")
        print(f"  Precision:        {precision:.4f}")
        print(f"  Recall:           {recall:.4f}")
        print(f"  F1-Score:         {f1:.4f}")
        print(f"  Avg Infer Time:   {avg_ms:.1f} ms/img")
        print(sep)

    def _print_video_results(self, miou, precision, recall, f1,
                             avg_fps, jitter, flicker, has_gt):
        sep = '=' * 45
        print(f"\n{sep}")
        print("🎬 Video Evaluation Results")
        print(sep)
        if has_gt:
            print(f"  Mean IoU:         {miou:.4f}")
            print(f"  Precision:        {precision:.4f}")
            print(f"  Recall:           {recall:.4f}")
            print(f"  F1-Score:         {f1:.4f}")
        print(f"  Avg FPS:          {avg_fps:.1f} frames/sec")
        print(f"  Temporal Jitter:  {jitter:.2f} px")
        print(f"  Flicker Rate:     {flicker:.2f} %")
        print(sep)
