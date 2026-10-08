import sys
import cv2
import numpy as np
from pathlib import Path
from models.base_model import BaseModel

# Ensure camera_detectors is in sys.path
_camera_dir = str(Path(__file__).parent.parent / "camera_detectors")
if _camera_dir not in sys.path:
    sys.path.insert(0, _camera_dir)


class CameraAdapter(BaseModel):
    """
    Adapter for Camera ROI detection algorithms (v3, v5, v6, v7).
    Detects camera bounds and active streams inside UI frames.
    """
    def __init__(self, version='v7', task='det'):
        self.version = str(version).lower()
        self.task = task
        self._video_detector = None
        
        self.is_video_mode = False
        
        if self.version == 'v7':
            from analyze_photos_v7_ultra import detect_camera_rois_v7, warm_up_v7
            try:
                warm_up_v7()
            except Exception as e:
                print(f"[CameraAdapter] Warmup v7 warning: {e}")
            self._single_fn = lambda img: detect_camera_rois_v7(img, use_downsample=False)
        elif self.version == 'v6':
            from analyze_photos_v6_ultra import detect_camera_rois_v6_ultra, VideoCameraDetector
            self._video_detector = VideoCameraDetector()
            self._single_fn = detect_camera_rois_v6_ultra
        elif self.version == 'v5':
            from analyze_photos_v5_translucent import detect_camera_rois_v5
            self._single_fn = detect_camera_rois_v5
        elif self.version == 'v3':
            from analyze_photos_v3 import detect_camera_rois
            self._single_fn = detect_camera_rois
        else:
            raise ValueError(f"Unsupported Camera ROI version: {version}. Choose from 'v7', 'v6', 'v5', 'v3'.")

    def reset(self):
        if self._video_detector is not None:
            self._video_detector.reset()

    def set_video_mode(self, is_video: bool):
        self.is_video_mode = is_video
        self.reset()

    def predict(self, image_path_or_array, conf_threshold=0.25):
        if isinstance(image_path_or_array, (str, Path)):
            arr = np.fromfile(str(image_path_or_array), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = image_path_or_array
            
        if img is None:
            return {'boxes': [], 'masks': [], 'classes': [], 'annotated_img': None}
            
        h, w = img.shape[:2]
        if self.is_video_mode and self._video_detector is not None:
            regions = self._video_detector.process_frame(img, detector_fn=self._single_fn)
        else:
            regions = self._single_fn(img)
        
        boxes = []
        masks = []
        for r in regions:
            boxes.append([r.x, r.y, r.x + r.w, r.y + r.h])
            m = np.zeros((h, w), dtype=np.uint8)
            # Clip bounds to image dimensions
            x1 = max(0, min(w, r.x))
            y1 = max(0, min(h, r.y))
            x2 = max(0, min(w, r.x + r.w))
            y2 = max(0, min(h, r.y + r.h))
            m[y1:y2, x1:x2] = 1
            masks.append(m)
            
        from analyze_photos_v3 import annotate_frame
        annotated_img = annotate_frame(img, regions)
        
        return {
            'boxes': boxes,
            'masks': masks,
            'classes': [0] * len(boxes),
            'annotated_img': annotated_img,
            'regions': regions
        }

    def predict_and_save(self, image_path, save_dir):
        preds = self.predict(image_path)
        if preds['annotated_img'] is not None:
            save_path = Path(save_dir) / Path(image_path).name
            ext = save_path.suffix.lower() or ".jpg"
            ok, buf = cv2.imencode(ext, preds['annotated_img'])
            if ok:
                buf.tofile(str(save_path))
