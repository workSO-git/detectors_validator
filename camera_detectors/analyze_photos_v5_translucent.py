# -*- coding: utf-8 -*-
"""
Camera ROI Detection - VERSION 5 EXPERIMENTAL (Translucent HUD Classifier)
========================================================================================
Extends V4 (PiDiNet Edge Model) with a targeted single-frame Translucent Overlay Filter.
  - Detects translucent HUD headers at y=39 (where underlying video stream texture bleeds through)
  - Distinguishes them from pitch-black solid header boxes (which remain cropped at y=104/168)
"""

import cv2
import numpy as np
import json
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

sys.path.insert(0, ".")
from analyze_photos_v3 import compute_iou, annotate_frame, CameraRegion
import analyze_photos_v4_pidinet as v4

def detect_camera_rois_v5(img: np.ndarray) -> List[CameraRegion]:
    # Start with baseline V4 detection
    rois = v4.detect_camera_rois_v4(img)
    main = rois[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    # Targeted Translucent Overlay Refinement for single-camera container (x=160, w=961 or x=64, w=1153)
    if main.y >= 95 and main.y <= 115 and main.w > 800:
        sub_gray = gray[:, main.x:main.x + main.w]
        sub_sat = hsv[:, main.x:main.x + main.w, 1]

        # Check if an outer UI header line exists at y in [35..45] or [55..65]
        edges_y = np.abs(cv2.Sobel(sub_gray, cv2.CV_64F, 0, 1, ksize=3))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(main.w * 0.40)), 1))
        lines_h = cv2.morphologyEx((edges_y > 35).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_h)
        line_dens = (lines_h > 0).mean(axis=1)

        # Check candidate outer top y (39 or 60)
        outer_candidates = [y for y in range(35, 65) if line_dens[y] > 0.15]
        if outer_candidates:
            outer_y = outer_candidates[0]
            # Inspect region between outer_y and current main.y (e.g. 39 to 104)
            span_std = float(sub_gray[outer_y:main.y].std())
            span_mean = float(sub_gray[outer_y:main.y].mean())
            span_sat = float(sub_sat[outer_y:main.y].mean())

            # If region between outer_y and main.y has active video stream texture
            # (span_std > 20.0, span_sat > 50.0, span_mean > 20.0), it is a TRANSLUCENT HUD overlay!
            if span_std > 20.0 and span_sat > 50.0 and span_mean > 20.0:
                main.h = main.h + (main.y - outer_y)
                main.y = outer_y

    rois = _apply_smart_telemetry_guard(rois, gray, h, w)
    return rois


def _apply_smart_telemetry_guard(rois: List[CameraRegion], gray: np.ndarray, h: int, w: int) -> List[CameraRegion]:
    for r in rois:
        if r.y >= 150 and r.y + r.h > 580:
            sub_w = max(10, r.w)
            sub_x1 = max(0, r.x)
            sub_x2 = min(w, r.x + sub_w)
            if sub_x2 > sub_x1 and h >= 600:
                spacer_means = gray[535:560, sub_x1:sub_x2].mean(axis=1)
                widget_stds = gray[560:min(h, 600), sub_x1:sub_x2].std(axis=1)
                if (spacer_means < 25.0).any() and (widget_stds > 30.0).any():
                    min_rel_y = int(np.argmin(spacer_means))
                    target_bot = 535 + min_rel_y
                    new_h = max(100, target_bot - r.y)
                    r.h = new_h
                    r.area = r.w * r.h
                    r.center_y = r.y + r.h / 2.0
    return rois

if __name__ == "__main__":
    print("Testing analyze_photos_v5_translucent.py...")
    with open("ground_truth/_ground_truth.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_dict = {item["file"]: item for item in gt_data}

    photo_dir = Path("photo")
    images = sorted([p for p in photo_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    ious = []
    pass_98 = 0
    errors = []

    for p in images:
        gt_item = gt_dict.get(p.name, {})
        gt_main = gt_item.get("main_roi", {})
        if not gt_main:
            continue

        arr = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        rois = detect_camera_rois_v5(img)
        main = rois[0]

        pred_box = {"x": main.x, "y": main.y, "w": main.w, "h": main.h}
        gt_box = {"x": gt_main["x"], "y": gt_main["y"], "w": gt_main["w"], "h": gt_main["h"]}

        iou = compute_iou(pred_box, gt_box)
        ious.append(iou)

        if iou >= 0.98:
            pass_98 += 1
        else:
            errors.append((p.name, iou, pred_box, gt_box))

    print("\n" + "=" * 75)
    print("V5 TRANSLUCENT HUD BENCHMARK RESULTS:")
    print("-" * 75)
    print(f"Total Ground Truth evaluated: {len(ious)}")
    print(f"Pass (IoU >= 0.98):           {pass_98} / {len(ious)} ({pass_98/len(ious)*100:.1f}%)")
    print(f"Mean IoU:                     {np.mean(ious)*100:.2f}%")
    print(f"Errors count:                 {len(errors)}")
    print("-" * 75)
    for name, iou, pred, gt in errors:
        print(f"  {name}:\n    IoU={iou:.3f} | Pred={pred} vs GT={gt}")
    print("=" * 75)
