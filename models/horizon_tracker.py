import cv2
import numpy as np
from math import radians, degrees, sin, cos, atan2, pi

class OpticalFlowHorizonTracker:
    def __init__(self, detect_fn, fov=60.0, max_frames=30, max_err=5.0, min_points=20):
        """
        detect_fn: Callable that takes (img_bgr) and returns (roll, pitch).
        """
        self.detect_fn = detect_fn
        self.fov = fov
        self.max_frames = max_frames
        self.min_points = min_points
        
        self.is_tracking = False
        self.frames_since_detect = 0
        self.tracked_points = None
        self.last_gray = None
        self.current_roll = 0.0
        self.current_pitch = 0.0

    def process_frame(self, img_bgr):
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if not self.is_tracking or self.frames_since_detect >= self.max_frames:
            self._reinit(img_bgr, gray, h, w)
            return self.current_roll, self.current_pitch

        # Optical Flow tracking
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            self.last_gray, gray, self.tracked_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if p1 is not None:
            good_mask = (status.flatten() == 1)
            good_new = p1[good_mask]
            good_old = self.tracked_points[good_mask]
            
            if len(good_new) < self.min_points:
                self._reinit(img_bgr, gray, h, w)
                return self.current_roll, self.current_pitch

            # Estimate global affine transform (rotation + translation + uniform scale)
            # RANSAC naturally ignores the static OSD text
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            
            if M is not None:
                # We have the transform M from previous frame to current frame
                # Let's apply this transform to our current horizon line
                
                # 1. Create two points on the current horizon line
                roll_rad = radians(self.current_roll)
                distance = self.current_pitch / self.fov * h
                angle_perp = roll_rad + pi / 2
                x_perp = distance * cos(angle_perp) + w / 2
                y_perp = distance * sin(angle_perp) + h / 2
                
                vx, vy = cos(roll_rad), sin(roll_rad)
                p_line1 = np.array([[x_perp - vx * 1000, y_perp - vy * 1000]], dtype=np.float32)
                p_line2 = np.array([[x_perp + vx * 1000, y_perp + vy * 1000]], dtype=np.float32)
                
                # Apply affine transform to the two points
                pts = np.vstack([p_line1, p_line2]).reshape(-1, 1, 2)
                pts_transformed = cv2.transform(pts, M).reshape(-1, 2)
                
                p1_t, p2_t = pts_transformed[0], pts_transformed[1]
                
                # Recalculate new roll and pitch from the transformed points
                dx = p2_t[0] - p1_t[0]
                dy = p2_t[1] - p1_t[1]
                
                new_roll_rad = atan2(dy, dx)
                self.current_roll = degrees(new_roll_rad) % 360
                
                # Pitch from distance to center
                cx, cy = w / 2, h / 2
                # Line equation: (x - p1x)*dy - (y - p1y)*dx = 0
                # Normal vector: (-dy, dx). Wait, length of (dx, dy) might have changed due to scale.
                length = np.hypot(dx, dy)
                if length > 1e-5:
                    nx, ny = -dy / length, dx / length
                    dist_pixels = (cx - p1_t[0]) * nx + (cy - p1_t[1]) * ny
                    self.current_pitch = -(dist_pixels / h) * self.fov
                
                # Update tracking state
                # Re-find features only if points drop or every 10 frames to save time
                if len(good_new) < 100 or self.frames_since_detect % 10 == 0:
                    p0 = cv2.goodFeaturesToTrack(gray, mask=getattr(self, 'osd_mask', None), maxCorners=300, qualityLevel=0.01, minDistance=10)
                    if p0 is not None and len(p0) > self.min_points:
                        self.tracked_points = p0
                    else:
                        self.tracked_points = good_new.reshape(-1, 1, 2)
                else:
                    self.tracked_points = good_new.reshape(-1, 1, 2)
                    
                self.last_gray = gray
                self.frames_since_detect += 1
            else:
                self._reinit(img_bgr, gray, h, w)
        else:
            self._reinit(img_bgr, gray, h, w)

        return self.current_roll, self.current_pitch

    def _reinit(self, img_bgr, gray, h, w):
        roll, pitch = self.detect_fn(img_bgr)
        if roll is None:
            self.is_tracking = False
            return
            
        self.current_roll = roll
        self.current_pitch = pitch
        
        # Track features over the whole image, avoiding static corners and center (OSD text)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        margin_y, margin_x = int(h * 0.15), int(w * 0.20)
        mask[:margin_y, :] = 0
        mask[-margin_y:, :] = 0
        mask[:, :margin_x] = 0
        mask[:, -margin_x:] = 0
        
        # Mask out center crosshair/artificial horizon
        cv2.rectangle(mask, (w//2 - 100, h//2 - 100), (w//2 + 100, h//2 + 100), 0, -1)
        
        # Reusable property for process_frame to use the same mask
        self.osd_mask = mask
        
        p0 = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=300, qualityLevel=0.01, minDistance=10)
        
        if p0 is not None and len(p0) >= self.min_points:
            self.tracked_points = p0
            self.last_gray = gray
            self.frames_since_detect = 0
            self.is_tracking = True
        else:
            self.is_tracking = False
