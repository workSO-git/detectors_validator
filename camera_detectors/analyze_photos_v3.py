# -*- coding: utf-8 -*-
"""
Camera ROI Detection - VERSION 3 (Mask-Based & Structural Analysis)
=====================================================================
Key features:
  1. Monotonicity Mask (M_mono): Identifies flat, low-variance regions (black/grey/white borders, HUD bars, blank feeds)
  2. Structural Line Mask (M_lines): Identifies sharp horizontal and vertical UI dividing lines
  3. Active Content Extraction: Combines masks to locate true video stream boundaries (top, bottom, left, right)
  4. Adaptive Multi-Camera Detection:
     - Detects split-screens (left/right split at x=640, 2x2 grid, custom 3-cam layouts)
     - Evaluates spatial activity/entropy (std > 10) for each candidate cell
     - Automatically selects the most active camera as MAIN (handles blank/monotonic left feeds)
  5. Exports diagnostic masks to `masks/` folder
  6. Evaluates Mean IoU against Ground Truth (`ground_truth/_ground_truth.json`)
"""

import cv2
import numpy as np
import json
import sys
import io
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


@dataclass
class CameraRegion:
    x: int
    y: int
    w: int
    h: int
    area: int
    activity: float    # spatial standard deviation / entropy score
    center_x: float
    center_y: float
    rank: int          # 1=main (active), 2+=secondary


def compute_masks(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Monotonicity Mask, Structural Line Mask, Active Content Mask, and RGB Composite Mask.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Monotonicity Mask (local std dev < threshold or extreme uniform values)
    mean = cv2.blur(gray.astype(float), (5, 5))
    sqr_mean = cv2.blur((gray.astype(float)**2), (5, 5))
    local_var = np.maximum(0, sqr_mean - mean**2)
    local_std = np.sqrt(local_var)

    # Monotone pixels: low local texture (< 7.5) OR extreme dark (< 15) / bright (> 220) uniform regions
    mono_mask = ((local_std < 7.5) | (gray < 15) | (gray > 220)).astype(np.uint8) * 255

    # 2. Structural Lines Mask (Sobel gradients & morphological opening)
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    thresh_x = (sobel_x > 35).astype(np.uint8) * 255
    thresh_y = (sobel_y > 35).astype(np.uint8) * 255

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    lines_h = cv2.morphologyEx(thresh_y, cv2.MORPH_OPEN, kernel_h)
    lines_v = cv2.morphologyEx(thresh_x, cv2.MORPH_OPEN, kernel_v)
    lines_mask = cv2.bitwise_or(lines_h, lines_v)

    # 3. Active Content Mask
    interface_mask = cv2.bitwise_or(mono_mask, lines_mask)
    active_mask = cv2.bitwise_not(interface_mask)

    # 4. Translucent Tinted Mask Overlay:
    vis_mask = img.copy()
    tint_color = np.array([180, 50, 220], dtype=np.uint8)
    mono_bool = (mono_mask > 0)
    blended = cv2.addWeighted(img, 0.4, np.full_like(img, tint_color), 0.6, 0)
    vis_mask[mono_bool] = blended[mono_bool]

    lines_bool = (lines_mask > 0)
    vis_mask[lines_bool] = (0, 255, 255)

    return mono_mask, lines_mask, active_mask, vis_mask


def safe_max(arr: np.ndarray, default: float = 0.0) -> float:
    return float(arr.max()) if len(arr) > 0 else default

def safe_mean(arr: np.ndarray, default: float = 0.0) -> float:
    return float(arr.mean()) if len(arr) > 0 else default

def make_cum(arr: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(arr, dtype=np.float64), (1, 0))

def span_mean(cum: np.ndarray, start: int, end: int) -> float:
    if start >= end:
        return 0.0
    return float((cum[end] - cum[start]) / (end - start))

def refine_cell_y(img: np.ndarray, cell_x: int, cell_w: int, default_top: int = 0, default_bot: int = 0, pre_gray: np.ndarray = None, pre_sobel_y: np.ndarray = None, pre_mono_m: np.ndarray = None) -> Tuple[int, int]:
    """
    Refines vertical top_y and bot_y bounds inside a specific camera cell width span [cell_x, cell_x + cell_w].
    """
    h, w = img.shape[:2]
    sub_gray = pre_gray[:, cell_x:cell_x + cell_w] if pre_gray is not None else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, cell_x:cell_x + cell_w]
    row_stds = sub_gray.std(axis=1)
    row_means = sub_gray.mean(axis=1)

    sub_sobel_y = pre_sobel_y[:, cell_x:cell_x + cell_w] if pre_sobel_y is not None else np.abs(cv2.Sobel(sub_gray, cv2.CV_64F, 0, 1, ksize=3))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(cell_w * 0.40)), 1))
    lines_h = cv2.morphologyEx((sub_sobel_y > 35).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_h)
    h_line_dens = (lines_h > 0).mean(axis=1)

    if pre_mono_m is not None:
        sub_mono_m = pre_mono_m[:, cell_x:cell_x + cell_w]
    else:
        local_std = np.sqrt(np.maximum(0, cv2.blur(sub_gray.astype(float)**2, (5, 5)) - cv2.blur(sub_gray.astype(float), (5, 5))**2))
        sub_mono_m = ((local_std < 7.5) | (sub_gray < 15) | (sub_gray > 220)).astype(np.uint8)

    mono_row = sub_mono_m.mean(axis=1)

    cum_mono = make_cum(mono_row)
    cum_std = make_cum(row_stds)
    cum_mean = make_cum(row_means)

    top_y = default_top if default_top > 0 else 0
    best_top = -1.0
    best_line = 0.0

    for r in range(1, int(h * 0.35)):
        pre_std = span_mean(cum_std, 0, r) if r > 0 else row_stds[0]
        post_std = span_mean(cum_std, r, min(h, r + 20))
        line = h_line_dens[r]
        pre_mono = span_mean(cum_mono, max(0, r - 10), r) if r > 0 else mono_row[0]
        post_mono = span_mean(cum_mono, r, min(h, r + 20))
        mono_drop = max(0.0, pre_mono - post_mono)

        # Require real boundary markers (line, mono drop, or outer transition)
        if line < 0.05 and mono_drop < 0.12 and not (pre_std < 1.5 and post_std > 5.0):
            continue

        # Do not allow distant weak candidates to override strong global top_y
        if default_top > 50 and abs(r - default_top) > 45 and line < 0.40:
            continue

        if default_top > 50 and r < default_top - 25:
            continue

        below_mono = span_mean(cum_mono, r, min(h, r + 30))
        below_std = span_mean(cum_std, r, min(h, r + 30))
        
        # Outer header boundary condition (e.g. BRAVO 04-20 UI header at y=68)
        is_outer_header = (line > 0.30 and pre_mono > 0.85 and row_means[min(h-1, r+5)] > 70.0)

        # Skip candidate if region below row r is a flat monotone UI header bar (mono > 0.70) unless it is a true outer header
        if r < 150 and below_mono > 0.70 and not is_outer_header:
            continue

        # Ignore lineless top letterbox transition into UI header text at r < 35 if mono_drop is zero
        if r < 35 and line < 0.05 and mono_drop < 0.10 and post_mono > 0.50:
            continue

        # If previous best_top was inside a monotone dark header box (below_std < 10.0)
        # AND current row r has strong line (>0.20) and higher active texture (post_std > 20.0), let r take over!
        if best_top > 0 and top_y > 0 and top_y < r - 15:
            prev_is_outer_header = (best_line > 0.30 and span_mean(cum_mono, max(0, top_y - 10), top_y) > 0.85 and row_means[min(h-1, top_y+5)] > 70.0)
            if prev_is_outer_header:
                continue
            prev_below_std = span_mean(cum_std, top_y, min(h, top_y + 30))
            if (prev_below_std < 10.0) and line > 0.20 and post_std > 20.0:
                best_top = -1.0

        if (pre_mono > 0.40 or pre_std < 15.0) and (mono_drop > 0.15 or line > 0.02 or post_std > pre_std + 5.0):
            if best_top > 0 and span_mean(cum_mono, top_y, r) < 0.70 and span_mean(cum_mean, top_y, r) > 20.0 and span_mean(cum_std, top_y, r) > 4.0 and line < 0.20:
                continue
            score = mono_drop * 10.0 + line * 300.0 + (post_std - pre_std) * 0.5
            if score > best_top and score > 2.5:
                best_top = score
                top_y = r
                best_line = line

    bot_y = default_bot if default_bot > 0 else h
    best_bot = -1.0
    for r in range(h - 2, int(h * 0.60), -1):
        pre_std = span_mean(cum_std, max(0, r - 20), r)
        post_std = span_mean(cum_std, r, h) if r < h - 1 else row_stds[-1]
        line = h_line_dens[r]
        pre_mono = span_mean(cum_mono, max(0, r - 20), r)
        post_mono = span_mean(cum_mono, r, min(h, r + 10)) if r < h - 1 else mono_row[-1]
        mono_drop = max(0.0, post_mono - pre_mono)

        # Gap protection for HUD footer text bars (e.g. GOLF/OSKAR):
        if r > 600 and span_mean(cum_std, max(0, r - 8), r) < 3.0:
            best_bot = -1.0
            bot_y = r
            continue

        if (post_mono > 0.50 or post_std < 12.0) and (mono_drop > 0.20 or line > 0.05 or pre_std > post_std + 5.0):
            score = mono_drop * 10.0 + line * 200.0 + (pre_std - post_std) * 0.5
            if score > best_bot and score > 2.0:
                best_bot = score
                bot_y = r + 1

    return top_y, bot_y - top_y


def detect_camera_rois(img: np.ndarray, max_width: int = 99999, *args, **kwargs) -> List[CameraRegion]:
    orig_h, orig_w = img.shape[:2]
    if orig_w > max_width:
        scale = max_width / float(orig_w)
        img_proc = cv2.resize(img, (max_width, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        img_proc = img

    h, w = img_proc.shape[:2]
    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    
    row_stds = gray.std(axis=1)
    col_stds = gray.std(axis=0)

    kernel_h_25 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    lines_h_25 = cv2.morphologyEx((sobel_y > 35).astype(np.uint8)*255, cv2.MORPH_OPEN, kernel_h_25)
    h_line_dens_25 = (lines_h_25 > 0).mean(axis=1)

    kernel_h_40 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.40), 1))
    kernel_v_25 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.25)))
    lines_h = cv2.morphologyEx((sobel_y > 35).astype(np.uint8)*255, cv2.MORPH_OPEN, kernel_h_40)
    lines_v = cv2.morphologyEx((sobel_x > 35).astype(np.uint8)*255, cv2.MORPH_OPEN, kernel_v_25)

    h_line_dens = (lines_h > 0).mean(axis=1)
    v_line_dens = (lines_v > 0).mean(axis=0)
    
    local_std = np.sqrt(np.maximum(0, cv2.blur(gray.astype(float)**2, (5, 5)) - cv2.blur(gray.astype(float), (5, 5))**2))
    mono_m = ((local_std < 7.5) | (gray < 15) | (gray > 220)).astype(np.uint8)
    mono_row = mono_m.mean(axis=1)
    mono_col = mono_m.mean(axis=0)

    row_means = gray.mean(axis=1)
    cum_mono_row = make_cum(mono_row)
    cum_mono_col = make_cum(mono_col)
    cum_row_stds = make_cum(row_stds)
    cum_col_stds = make_cum(col_stds)
    cum_row_means = make_cum(row_means)

    # Side pillar check: detects dark monotone margins bounded by vertical UI lines (e.g. DARTS-16.04)
    v_line_dens_25 = (cv2.morphologyEx((sobel_x > 35).astype(np.uint8)*255, cv2.MORPH_OPEN, kernel_v_25) > 0).mean(axis=0)
    max_v_left = safe_max(v_line_dens_25[1:int(w*0.35)])
    max_v_right = safe_max(v_line_dens_25[int(w*0.65):w-1])
    left_mono = mono_col[:20].mean()
    right_mono = mono_col[-20:].mean()
    left_mean = gray[:, :20].mean()
    right_mean = gray[:, -20:].mean()

    has_side_pillars = (max_v_left > 0.40 or max_v_right > 0.40) and ((left_mono > 0.40 and left_mean < 35.0) or (right_mono > 0.40 and right_mean < 35.0))
    has_side_margins = (mono_col[:150].mean() > 0.50 and gray[:, :150].mean() < 35.0 and mono_col[-150:].mean() > 0.50 and gray[:, -150:].mean() < 35.0)

    # Check center vertical divider for split-screen detection
    kernel_v_25px = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    kernel_v_15px = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    lines_v_soft_25 = cv2.morphologyEx((sobel_x > 25).astype(np.uint8)*255, cv2.MORPH_OPEN, kernel_v_25px)
    lines_v_soft_15 = cv2.morphologyEx((sobel_x > 25).astype(np.uint8)*255, cv2.MORPH_OPEN, kernel_v_15px)

    center_v_soft_25 = float((lines_v_soft_25 > 0).mean(axis=0)[637:643].max()) if lines_v_soft_25.size > 0 else 0.0
    center_v_soft_15 = float((lines_v_soft_15 > 0).mean(axis=0)[637:643].max()) if lines_v_soft_15.size > 0 else 0.0

    is_true_divider = (center_v_soft_25 > 0.30) or (center_v_soft_15 > 0.32)

    # 0. Check Fullframe Stream
    top_white_bar = (gray[:25, :].mean() > 195 and mono_row[:25].mean() > 0.60)
    top_offset = 25 if top_white_bar else 0

    if w <= 700 and h <= 500:
        if row_stds[:15].mean() > 10.0 and row_stds[-15:].mean() > 10.0:
            return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(gray.std()), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]

    # Robust Fullframe Check for 1280x720 / 1232x720 screenshots with edge activity
    if w >= 1200 and h >= 700 and not has_side_pillars:
        if (mono_row[:15].mean() < 0.25 or top_white_bar) and mono_row[-15:].mean() < 0.25 and mono_col[:15].mean() < 0.25 and mono_col[-15:].mean() < 0.25:
            if safe_max(h_line_dens_25[:25]) < 0.05 and safe_max(h_line_dens_25[-25:]) < 0.05:
                return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(gray.std()), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]
        # Active texture on all 4 borders AND no split-screen divider (Charlie-16-59-42 fullframe fix)
        if not is_true_divider and col_stds[:20].mean() > 20.0 and col_stds[-20:].mean() > 20.0 and row_stds[:20].mean() > 10.0 and row_stds[-20:].mean() > 10.0:
            return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(gray.std()), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]
        if not is_true_divider and row_stds[:20].mean() > 12.0 and row_stds[-20:].mean() > 12.0 and col_stds[:20].mean() > 12.0 and col_stds[-20:].mean() > 12.0:
            if safe_max(h_line_dens_25[:20]) < 0.03 and safe_max(h_line_dens_25[-20:]) < 0.03:
                return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(gray.std()), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]

    # 1. LEFT BOUNDARY
    left_x = 0
    best_left_score = -1.0
    best_left_has_line = False
    for c in range(1, int(w * 0.35)):
        pre_std = span_mean(cum_col_stds, 0, c) if c > 0 else col_stds[0]
        post_std = span_mean(cum_col_stds, c, min(w, c+20))
        line = v_line_dens[c]
        pre_mono = span_mean(cum_mono_col, max(0, c-10), c) if c > 0 else mono_col[0]
        post_mono = span_mean(cum_mono_col, c, min(w, c+20))
        
        mono_drop = max(0.0, pre_mono - post_mono)
        if (pre_mono > 0.50 or pre_std < 12.0) and (mono_drop > 0.15 or line > 0.02 or post_std > pre_std + 5.0):
            if line < 0.01 and pre_std > 18.0:
                continue
            if best_left_score > 0 and best_left_has_line and line < 0.15:
                continue
            score = mono_drop * 10.0 + line * 300.0 + (post_std - pre_std) * 0.5
            if score > best_left_score and score > 1.0:
                best_left_score = score
                left_x = c
                best_left_has_line = (line > 0.15)

    # 2. RIGHT BOUNDARY
    right_x = w
    best_right_score = -1.0
    for c in range(w - 2, int(w * 0.60), -1):
        pre_std = span_mean(cum_col_stds, max(0, c-20), c)
        post_std = span_mean(cum_col_stds, c, w) if c < w-1 else col_stds[-1]
        line = v_line_dens[c]
        pre_mono = span_mean(cum_mono_col, max(0, c-20), c)
        post_mono = span_mean(cum_mono_col, c, min(w, c+10)) if c < w-1 else mono_col[-1]
        
        mono_drop = max(0.0, post_mono - pre_mono)
        # Rigid right boundary protection: require true UI line or monotonicity drop
        if post_std > 20.0 and pre_std > 20.0 and line < 0.10 and mono_drop < 0.10:
            continue

        if (post_mono > 0.50 or post_std < 12.0) and (mono_drop > 0.15 or line > 0.02 or pre_std > post_std + 5.0):
            score = mono_drop * 10.0 + line * 200.0 + (pre_std - post_std) * 0.5
            if score > best_right_score and score > 1.0:
                best_right_score = score
                right_x = c + 1

    # 3. TOP BOUNDARY
    top_y = 0
    best_top_score = -1.0
    best_top_line = 0.0

    for r in range(1, int(h * 0.35)):
        pre_std = span_mean(cum_row_stds, 0, r) if r > 0 else row_stds[0]
        post_std = span_mean(cum_row_stds, r, min(h, r+20))
        line = h_line_dens[r]
        pre_mono = span_mean(cum_mono_row, max(0, r-10), r) if r > 0 else mono_row[0]
        post_mono = span_mean(cum_mono_row, r, min(h, r+20))
        
        mono_drop = max(0.0, pre_mono - post_mono)
        is_outer_trans = (pre_std < 1.5 and post_std > 2.5 and r >= 35)

        # REJECT false boundaries in active content (require line, mono drop, or outer transition)
        if line < 0.05 and mono_drop < 0.12 and not is_outer_trans:
            continue

        # Reject lineless/weak top-edge window frame noise at r < 30 if region below is still monotone background
        if r < 30 and line < 0.25 and post_mono > 0.70:
            continue

        # Skip candidate if region below row r is a flat monotone UI header bar (mono > 0.70) unless line >= 0.30 or mono_drop >= 0.12
        if r < 150 and line < 0.30 and mono_drop < 0.12 and span_mean(cum_mono_row, r, min(h, r+30)) > 0.70:
            continue

        # Outer UI header protection: once valid outer UI header locked, ignore deeper scene horizon lines
        if top_y > 0 and best_top_line > 0.30 and span_mean(cum_mono_row, max(0, top_y - 10), top_y) > 0.85 and gray[min(h-1, top_y+5), :].mean() > 70.0 and r > top_y + 15:
            continue

        if (pre_mono > 0.40 or pre_std < 15.0 or is_outer_trans) and (mono_drop > 0.15 or line > 0.02 or post_std > pre_std + 3.0 or is_outer_trans):
            # Prefer strong structural UI lines over line-less margin transitions:
            if best_top_score > 0 and best_top_line > 0.15 and line < 0.15:
                continue
            # Horizon protection for fuzzy/thick multi-row edge bands:
            is_thick_line_band = (sum(h_line_dens_25[max(0, r-2):min(h, r+3)] > 0.05) >= 4)
            if is_thick_line_band and pre_mono < 0.90:
                continue
            # Discard line candidates < 50% screen coverage if region above is non-monotone active content
            if pre_mono < 0.70 and line < 0.50:
                continue
            # Strong structural lines (>0.35 coverage) can win as top boundary
            if line < 0.35:
                if best_top_score > 0 and top_y >= 35 and span_mean(cum_row_stds, top_y, r) > 3.5:
                    continue
                if best_top_score > 0 and top_y >= 35 and span_mean(cum_row_means, top_y, r) > 20.0 and span_mean(cum_row_stds, top_y, r) > 4.0 and line < 0.20:
                    continue

            score = mono_drop * 10.0 + line * 300.0 + (post_std - pre_std) * 0.5
            if is_outer_trans:
                score += 15.0

            if score > best_top_score and score > 1.0:
                best_top_score = score
                top_y = r
                best_top_line = line

    # 4. BOTTOM BOUNDARY
    bot_y = h
    best_bot_score = -1.0
    for r in range(h - 2, int(h * 0.60), -1):
        pre_std = span_mean(cum_row_stds, max(0, r-20), r)
        post_std = span_mean(cum_row_stds, r, h) if r < h-1 else row_stds[-1]
        line = h_line_dens[r]
        pre_mono = span_mean(cum_mono_row, max(0, r-20), r)
        post_mono = span_mean(cum_mono_row, r, min(h, r+10)) if r < h-1 else mono_row[-1]
        mono_drop = max(0.0, post_mono - pre_mono)
        
        # Gap protection for HUD footer text bars (e.g. GOLF/OSKAR):
        if r > 600 and span_mean(cum_row_stds, max(0, r - 8), r) < 3.0:
            best_bot_score = -1.0
            bot_y = r
            continue

        if (post_mono > 0.25 or post_std < 15.0) and span_mean(cum_row_stds, r, min(h, r+20)) < 20.0 and (mono_drop > 0.20 or line > 0.05 or pre_std > post_std + 5.0):
            score = mono_drop * 10.0 + line * 200.0 + (pre_std - post_std) * 0.5
            if score > best_bot_score and score > 2.0:
                best_bot_score = score
                bot_y = r + 1

    # -------------------------------------------------------------
    # BRANCH 2: LAYOUT_MULTI_CAM (Split-Screen / Multi-Camera Grid)
    # -------------------------------------------------------------
    if (right_x - left_x) > 800 and is_true_divider:
        div_col = (lines_v_soft_15[:, 637:643] > 0).any(axis=1)
        div_indices = np.where(div_col)[0]
        valid_div = [idx for idx in div_indices if idx >= 45]
        if valid_div:
            top_y = int(valid_div[0])

        l_top_y, l_h = refine_cell_y(img, left_x, 640 - left_x, default_top=top_y, default_bot=bot_y, pre_gray=gray, pre_sobel_y=sobel_y, pre_mono_m=mono_m)
        r_top_y, r_h = refine_cell_y(img, 640, right_x - 640, default_top=top_y, default_bot=bot_y, pre_gray=gray, pre_sobel_y=sobel_y, pre_mono_m=mono_m)

        left_act = float(gray[l_top_y:l_top_y+l_h, left_x:640].std())
        right_act = float(gray[r_top_y:r_top_y+r_h, 640:right_x].std())
        
        cell_left = CameraRegion(x=left_x, y=l_top_y, w=640-left_x, h=l_h, area=(640-left_x)*l_h, activity=left_act, center_x=(left_x+640)/2.0, center_y=l_top_y+l_h/2.0, rank=1)
        cell_right = CameraRegion(x=640, y=r_top_y, w=right_x-640, h=r_h, area=(right_x-640)*r_h, activity=right_act, center_x=(640+right_x)/2.0, center_y=r_top_y+r_h/2.0, rank=2)
        
        # Select RIGHT cell as MAIN if LEFT cell is inactive (left_act < 15) or RIGHT is overwhelmingly dominant (diff > 50)
        if (left_act < 15.0 and right_act > 30.0) or (right_act > left_act + 50.0):
            cell_right.rank = 1
            cell_left.rank = 2
            return [cell_right, cell_left]
        else:
            return [cell_left, cell_right]

    # -------------------------------------------------------------
    # BRANCH 3: LAYOUT_SINGLE_CAM (Single Camera Bounded Stream)
    # -------------------------------------------------------------
    act = float(gray[top_y:bot_y, left_x:right_x].std())
    main_reg = CameraRegion(x=left_x, y=top_y, w=right_x - left_x, h=bot_y - top_y, area=(right_x-left_x)*(bot_y-top_y), activity=act, center_x=(left_x+right_x)/2.0, center_y=(top_y+bot_y)/2.0, rank=1)
    return [main_reg]


# ─────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────

def draw_corner_brackets(img, region: CameraRegion, color=(0, 0, 255), thickness=4, arm_ratio=0.10):
    x, y, w, h = region.x, region.y, region.w, region.h
    arm = max(20, int(min(w, h) * arm_ratio))
    cv2.line(img, (x, y), (x + arm, y), color, thickness)
    cv2.line(img, (x, y), (x, y + arm), color, thickness)
    bx, by = x + w, y + h
    cv2.line(img, (bx, by), (bx - arm, by), color, thickness)
    cv2.line(img, (bx, by), (bx, by - arm), color, thickness)
    return img


def annotate_frame(img: np.ndarray, regions: List[CameraRegion]) -> np.ndarray:
    vis = img.copy()
    main = regions[0]

    for r in regions[1:]:
        cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 165, 255), 2)
        cv2.putText(vis, f"cam#{r.rank} {r.w}x{r.h}", (r.x + 5, r.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    draw_corner_brackets(vis, main, color=(0, 0, 255), thickness=4)
    cx, cy = int(main.center_x), int(main.center_y)
    cv2.line(vis, (cx - 25, cy), (cx + 25, cy), (0, 0, 255), 2)
    cv2.line(vis, (cx, cy - 25), (cx, cy + 25), (0, 0, 255), 2)
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

    lbl = f"MAIN {main.w}x{main.h} @ ({main.x},{main.y})  center=({main.center_x:.0f},{main.center_y:.0f})"
    ty = main.y + 28 if main.y + 28 < vis.shape[0] else main.y + main.h - 10
    cv2.putText(vis, lbl, (main.x + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return vis


# ─────────────────────────────────────────────────────────────────
# Benchmark & Evaluation against Ground Truth
# ─────────────────────────────────────────────────────────────────

def compute_iou(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    x1, y1, w1, h1 = box1["x"], box1["y"], box1["w"], box1["h"]
    x2, y2, w2, h2 = box2["x"], box2["y"], box2["w"], box2["h"]

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def process_and_evaluate(photo_dir="photo", mask_dir="masks", out_dir="roi_detected", gt_json="ground_truth/_ground_truth.json"):
    photo_dir = Path(photo_dir)
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    gt_json = Path(gt_json)

    mask_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    gt_dict = {}
    if gt_json.exists():
        with open(gt_json, encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                gt_dict[item["file"]] = item

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = sorted([p for p in photo_dir.iterdir() if p.suffix.lower() in image_exts])

    print(f"Running Master ROI Detection on {len(images)} images...")
    print(f"Masks output -> '{mask_dir}/'")
    print(f"Detected ROI output -> '{out_dir}/'")
    print("-" * 75)

    v3_results = []
    iou_scores = []
    exact_matches = 0
    start_time = time.time()

    for img_path in images:
        arr = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        mono_m, lines_m, active_m, vis_m = compute_masks(img)
        regions = detect_camera_rois(img)
        main = regions[0]

        vis_mask_annotated = annotate_frame(vis_m, regions)
        ext = img_path.suffix.lower() or ".png"
        ok1, enc1 = cv2.imencode(ext, vis_mask_annotated)
        if ok1:
            enc1.tofile(str(mask_dir / f"mask_{img_path.name}"))

        vis_photo_annotated = annotate_frame(img, regions)
        ok2, enc2 = cv2.imencode(ext, vis_photo_annotated)
        if ok2:
            enc2.tofile(str(out_dir / img_path.name))

        gt_item = gt_dict.get(img_path.name, {})
        gt_main = gt_item.get("main_roi", {})
        iou = 0.0
        if gt_main:
            pred_box = {"x": main.x, "y": main.y, "w": main.w, "h": main.h}
            iou = compute_iou(pred_box, gt_main)
            iou_scores.append(iou)
            if iou > 0.95:
                exact_matches += 1

        print(f"  [{img_path.name}] ROI: {main.w}x{main.h} @ ({main.x},{main.y}) | IoU: {iou:.3f}")

        v3_results.append({
            "file": img_path.name,
            "container": {"w": w, "h": h},
            "cameras_detected": len(regions),
            "main_roi": {
                "x": main.x, "y": main.y, "w": main.w, "h": main.h,
                "center_x": main.center_x, "center_y": main.center_y,
            },
            "all_regions": [
                {"rank": r.rank, "x": r.x, "y": r.y, "w": r.w, "h": r.h, "activity": round(r.activity, 1)}
                for r in regions
            ],
            "iou_vs_gt": round(iou, 4),
        })

    total_time = time.time() - start_time
    avg_fps = len(images) / max(total_time, 0.001)
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0

    summary_path = out_dir / "_results_v3.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(v3_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(f"ALGORITHM PERFORMANCE SUMMARY:")
    print(f"  Total Images Processed: {len(v3_results)}")
    print(f"  Total Time:             {total_time:.2f} s  ({avg_fps:.1f} FPS)")
    print(f"  Mean IoU vs GT:         {mean_iou:.4f}  ({mean_iou*100:.1f}%)")
    print(f"  Exact Matches (IoU>0.95): {exact_matches}/{len(iou_scores)} ({exact_matches/max(1,len(iou_scores))*100:.1f}%)")
    print("=" * 75)

    return v3_results, mean_iou


if __name__ == "__main__":
    process_and_evaluate()
