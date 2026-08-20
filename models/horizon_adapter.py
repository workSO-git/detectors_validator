import sys
import os
import cv2
import numpy as np
from math import radians, cos, sin, pi

# Append the HorizonDetection directory to sys.path so we can import it
HORIZON_DIR = r"C:\Users\Sasha\projects\CV\HorizonDetection\HorizonDetection"
if HORIZON_DIR not in sys.path:
    sys.path.insert(0, HORIZON_DIR)

from core.find_horizon import HorizonDetector
from video_runner import EXCLUSION_THRESH, FOV, ACCEPTABLE_VARIANCE

class HorizonAdapter:
    """
    Adapter for the traditional OpenCV-based HorizonDetector to match the expected API.
    """
    def __init__(self, task='seg', video_mode=True):
        self.task = task
        # video_mode=True: keeps temporal state between frames (better for video streaming).
        # video_mode=False: resets state each frame (safer for shuffled dataset evaluation).
        self.video_mode = video_mode
        self.detector = HorizonDetector(EXCLUSION_THRESH, FOV, ACCEPTABLE_VARIANCE, frame_shape=(100, 100))
        self.last_h = 1080

    def predict(self, img_bgr: np.ndarray):
        """
        img_bgr: original BGR image.
        Returns a list of binary masks (for task='seg').
        """
        h, w = img_bgr.shape[:2]
        
        if h != self.last_h:
            # We initialize the detector with (100, 100) because we will crop_and_scale
            # the image to 100x100 before passing it in.
            self.detector = HorizonDetector(EXCLUSION_THRESH, FOV, ACCEPTABLE_VARIANCE, frame_shape=(100, 100))
            self.last_h = h

        # The original find_horizon algorithm is heavily tuned for 100x100 inputs
        # (e.g., POOLING_KERNEL_SIZE=5, min points = 12). Passing the full HD frame
        # breaks its contour clustering logic and causes it to return None.
        # We must use its native crop_and_scale to feed it a 100x100 square.
        inference_res = (100, 100)
        from core.crop_and_scale import get_cropping_and_scaling_parameters, crop_and_scale
        params = get_cropping_and_scaling_parameters((w, h), inference_res)
        img_small = crop_and_scale(img_bgr, **params)
        
        # In video_mode, keep temporal state so the algorithm can track the horizon
        # stably across frames (prevents flipping). For shuffled dataset images, reset state.
        if not self.video_mode:
            self.detector.predicted_roll = None
            self.detector.predicted_pitch = None
            self.detector.recent_horizons = [None, None]

        roll, pitch, variance, is_good, _ = self.detector.find_horizon(img_small, diagnostic_mode=False)

        if roll is None or not is_good:
            # We still want to draw the HUD so the user can see WHY it didn't draw the mask (e.g., low confidence)
            if variance is not None:
                confidence = max(0.0, 100.0 - variance * (100.0 / ACCEPTABLE_VARIANCE))
                cv2.putText(img_bgr, f"Conf (Var): {confidence:.1f}% ({variance:.2f}) - HIDDEN", (20, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            return {
                'masks': [np.zeros((h, w), dtype=np.uint8)],
                'boxes': [],
                'classes': [0]
            }

        # Draw HUD with Roll, Pitch and Confidence (Variance)
        # We can calculate a pseudo-confidence from variance (lower variance = higher confidence)
        confidence = max(0.0, 100.0 - variance * (100.0 / ACCEPTABLE_VARIANCE)) if variance is not None else 0.0
        
        cv2.putText(img_bgr, f"Roll: {int(np.round(roll))}", (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"Pitch: {int(np.round(pitch))}", (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, f"Conf (Var): {confidence:.1f}% ({variance:.2f})", (20, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1, cv2.LINE_AA)

        # Generate sky mask using the full resolution (h, w)
        mask = self._generate_sky_mask(h, w, roll, pitch, FOV)
        return {
            'masks': [mask],
            'boxes': [],
            'classes': [0]
        }

    def _generate_sky_mask(self, h, w, roll_deg, pitch_deg, fov):
        roll = radians(roll_deg)
        sky_is_up = (roll >= 2*pi * 0.75) or (roll > 0 and roll <= 2*pi * 0.25)
        
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
