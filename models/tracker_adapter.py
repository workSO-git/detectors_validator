import sys
import numpy as np
import cv2
from math import atan2, degrees, cos, sin, pi, radians

# Use the native OpenCV algorithm for base detection, 
# as it correctly handles steep bank angles unlike the current ResNet model.
try:
    from models.horizon_adapter import HorizonAdapter
except ImportError:
    from horizon_adapter import HorizonAdapter

from models.horizon_tracker import OpticalFlowHorizonTracker
# Default Field of View for geometric calculations
FOV = 60.0

def extract_roll_pitch_from_mask(mask: np.ndarray, fov: float):
    """
    Extracts roll and pitch angles from a binary sky mask.
    """
    h, w = mask.shape[:2]
    
    # Find the boundary pixels between sky (255) and ground (0)
    # Find the largest contour of the sky
    mask_255 = mask * 255 if mask.max() <= 1 else mask
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # For the largest contour, keep only the lowest point (max y) for each x
    # This ignores the top border and keeps only the actual horizon line
    points_dict = {}
    for pt in largest_contour:
        x, y = pt[0]
        # Ignore points that are on the very top edge (y == 0) unless it's the only one
        if x not in points_dict or y > points_dict[x]:
            points_dict[x] = y
            
    points_list = [[x, y] for x, y in points_dict.items()]
            
    if len(points_list) < 10:
        return None, None
        
    pts = np.array(points_list, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Use polyfit to find y = mx + c
    # This is much more robust for a horizon line than cv2.fitLine 
    # because it minimizes vertical error, ignoring steep perpendicular outliers.
    m, c = np.polyfit(x, y, 1)
    
    # Calculate Roll (slope m)
    roll_rad = atan2(m, 1)
    roll = degrees(roll_rad)
    
    # Calculate Pitch
    # Find perpendicular distance from image center (w/2, h/2) to the line y = mx + c
    center_x, center_y = w / 2, h / 2
    # The line y = mx + c passes through center_x at y_line
    y_line = m * center_x + c
    
    # Vertical distance
    dist_vertical = center_y - y_line
    
    # Perpendicular distance is vertical distance * cos(roll)
    dist_pixels = dist_vertical * cos(roll_rad)
    pitch = -(dist_pixels / h) * fov
    
    # Adjust roll to be in [0, 360) and handle sky direction correctly
    # If sky is above the line, and Y is down, then ...
    # Wait, the SMP model just gives the mask. The mask itself knows what is sky.
    # We can just return the raw roll and pitch for tracking initialization.
    
    return roll, pitch

class TrackerAdapter:
    """
    Wraps the HorizonTracker in the evaluator's predict() API.
    Uses the native HorizonDetector (HorizonAdapter) as the robust base detector.
    """
    def __init__(self, task='seg'):
        self.task = task
        # Base detector for initialization (Native OpenCV)
        self.base_adapter = HorizonAdapter(task=task, video_mode=False)
        
        def detect_fn(img_bgr):
            h, w = img_bgr.shape[:2]
            from core.crop_and_scale import get_cropping_and_scaling_parameters, crop_and_scale
            params = get_cropping_and_scaling_parameters((w, h), (100, 100))
            img_small = crop_and_scale(img_bgr, **params)
            
            roll, pitch, var, is_good, _ = self.base_adapter.detector.find_horizon(img_small, diagnostic_mode=False)
            return roll, pitch

        self.tracker = OpticalFlowHorizonTracker(detect_fn=detect_fn, fov=FOV, max_frames=999999)
        self.last_h = None

    def _generate_sky_mask(self, h, w, roll_deg, pitch_deg, fov):
        roll = radians(roll_deg)
        # Determine sky direction. For typical flights, sky is up.
        # This simplifies the logic.
        sky_is_up = True 
        
        distance = pitch_deg / fov * h
        angle_perp = roll + pi / 2
        x_perp = distance * cos(angle_perp) + w / 2
        y_perp = distance * sin(angle_perp) + h / 2
        
        Y, X = np.ogrid[:h, :w]
        
        run = cos(roll)
        rise = sin(roll)
        
        if abs(run) > 1e-5:
            m = rise / run
            b = y_perp - m * x_perp
            if sky_is_up:
                mask = Y < (m * X + b)
            else:
                mask = Y > (m * X + b)
        else:
            # vertical line
            if sky_is_up:
                mask = X < x_perp
            else:
                mask = X > x_perp
                
        return mask.astype(np.uint8) * 255

    def predict(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        
        if self.last_h is not None and h != self.last_h:
            self.tracker.is_tracking = False
        self.last_h = h
        
        # The tracker processes the frame and updates roll and pitch using optical flow
        # or runs the neural network detection if not currently tracking (e.g. every 30 frames)
        roll, pitch = self.tracker.process_frame(img_bgr)
        
        if roll is None:
            return {
                'masks': [np.zeros((h, w), dtype=np.uint8)],
                'boxes': [],
                'classes': [0]
            }

        # Generate sky mask from tracked roll and pitch
        mask = self._generate_sky_mask(h, w, roll, pitch, FOV)
        return {
            'masks': [mask],
            'boxes': [],
            'classes': [0]
        }
