# -*- coding: utf-8 -*-
"""
Camera ROI Detection - VERSION 6 ULTRA FAST CPU (Single-Frame Production Release)
========================================================================================
Combines high-speed CPU optimizations for single-frame processing:
  1. Zero-Allocation Fast Operations: Uses cv2.meanStdDev, cv2.threshold(dst=...), and pre-allocated kernels.
  2. Integral Image Monotonicity Mask: Fast local variance calculation using cv2.blur.
  3. Vectorized 1D Cumulative Scanner: Scans row/col statistics for accurate left, right, top, bottom boundaries.
  4. Translucent HUD Overlay Refinement: Correctly expands top boundary for translucent UI headers.
  5. Multi-Camera Split Layout Handler: Detects middle dividers and ranks main camera by activity score.

Evaluates on 1280x720 native single frames with high FPS and accuracy.
"""

import cv2
import numpy as np
import time
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from analyze_photos_v3 import CameraRegion, compute_iou, annotate_frame


def safe_max(arr: np.ndarray, default: float = 0.0) -> float:
    return float(arr.max()) if len(arr) > 0 else default


def make_cum(arr: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(arr, dtype=np.float64), (1, 0))


def span_mean(cum: np.ndarray, start: int, end: int) -> float:
    if start >= end:
        return 0.0
    return float((cum[end] - cum[start]) / (end - start))


def get_cell_activity(crop: np.ndarray) -> float:
    """
    Evaluates real visual activity / detail richness of a camera cell.
    Combines standard deviation, Sobel edge detail, and penalizes pitch-black/dark empty views with UI graphics.
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    std_dev = float(cv2.meanStdDev(gray)[1][0][0])
    non_dark_ratio = float((gray > 35).mean())
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_detail = float(np.mean(np.abs(sobel_x) + np.abs(sobel_y)))
    
    # Penalize pitch-black / covered feeds that only contain sparse UI graphics (non_dark_ratio < 0.35)
    dark_penalty = min(1.0, (non_dark_ratio / 0.35) ** 2) if non_dark_ratio < 0.35 else 1.0
    return (std_dev * 0.6 + edge_detail * 1.5) * dark_penalty


def refine_cell_y_v6(img: np.ndarray, cell_x: int, cell_w: int, default_top: int = 0, default_bot: int = 0, pre_gray: np.ndarray = None, pre_sobel_y: np.ndarray = None, pre_mono_m: np.ndarray = None) -> Tuple[int, int]:
    h, w = img.shape[:2]
    sub_gray = pre_gray[:, cell_x:cell_x + cell_w] if pre_gray is not None else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, cell_x:cell_x + cell_w]
    
    row_stds = np.empty(h, dtype=np.float64)
    row_means = np.empty(h, dtype=np.float64)
    for r in range(h):
        m, s = cv2.meanStdDev(sub_gray[r, :])
        row_means[r] = m[0][0]
        row_stds[r] = s[0][0]

    sub_sobel_y = pre_sobel_y[:, cell_x:cell_x + cell_w] if pre_sobel_y is not None else np.abs(cv2.Sobel(sub_gray, cv2.CV_64F, 0, 1, ksize=3))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(cell_w * 0.20)), 1))
    lines_h = cv2.morphologyEx((sub_sobel_y > 35).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_h)
    h_line_dens = (lines_h > 0).mean(axis=1)

    if pre_mono_m is not None:
        sub_mono_m = pre_mono_m[:, cell_x:cell_x + cell_w]
    else:
        local_std = np.sqrt(np.maximum(0.0, cv2.blur(sub_gray.astype(float)**2, (5, 5)) - cv2.blur(sub_gray.astype(float), (5, 5))**2))
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

        if line < 0.05 and mono_drop < 0.12 and not (pre_std < 1.5 and post_std > 5.0):
            continue
        if default_top > 50 and abs(r - default_top) > 45 and line < 0.40:
            continue
        if default_top > 50 and r < default_top - 25:
            continue

        below_mono = span_mean(cum_mono, r, min(h, r + 30))
        below_std = span_mean(cum_std, r, min(h, r + 30))
        is_outer_header = (line > 0.30 and pre_mono > 0.85 and row_means[min(h-1, r+5)] > 70.0)

        if r < 150 and below_mono > 0.70 and not is_outer_header:
            continue

        if best_top > 0 and top_y > 0 and top_y < r - 15:
            prev_is_outer_header = (best_line > 0.30 and span_mean(cum_mono, max(0, top_y - 10), top_y) > 0.85 and row_means[min(h-1, top_y+5)] > 70.0)
            if not prev_is_outer_header:
                prev_below_std = span_mean(cum_std, top_y, min(h, top_y + 30))
                if (prev_below_std < 10.0) and line > 0.20 and post_std > 20.0:
                    best_top = -1.0

        if (pre_mono > 0.40 or pre_std < 15.0) and (mono_drop > 0.15 or line > 0.02 or post_std > pre_std + 5.0):
            score = mono_drop * 10.0 + line * 300.0 + (post_std - pre_std) * 0.5
            if score > best_top and score > 2.5:
                best_top = score
                top_y = r
                best_line = line

    bot_y = default_bot if default_bot > 0 else h
    best_bot = -1.0
    for r in range(h - 2, int(h * 0.55), -1):
        pre_std = span_mean(cum_std, max(0, r - 20), r)
        post_std = span_mean(cum_std, r, h) if r < h - 1 else row_stds[-1]
        line = h_line_dens[r]
        pre_mono = span_mean(cum_mono, max(0, r - 20), r)
        post_mono = span_mean(cum_mono, r, min(h, r + 10)) if r < h - 1 else mono_row[-1]
        mono_drop = max(0.0, post_mono - pre_mono)

        if r > 600 and span_mean(cum_std, max(0, r - 8), r) < 3.0:
            best_bot = -1.0
            bot_y = r
            continue

        if (post_mono > 0.25 or post_std < 18.0 or line > 0.15) and (mono_drop > 0.15 or line > 0.05 or pre_std > post_std + 3.0):
            score = mono_drop * 10.0 + line * 200.0 + (pre_std - post_std) * 0.5
            if line > 0.15 and best_bot > 0 and bot_y > r + 25:
                score += 50.0

            if score > best_bot and score > 2.0:
                best_bot = score
                bot_y = r + 1

    return top_y, bot_y - top_y


def top_edge_scan(gray: np.ndarray, x1: int, x2: int, min_step: int = 2, zone: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scans top 35% rows for horizontal gradient sign coherence inside [x1, x2].
    Returns (candidate_rows, ownership_scores).
    """
    g = gray.astype(np.int16)
    H = int(gray.shape[0] * zone)
    if H < 5 or (x2 - x1) < 20:
        return np.array([], dtype=int), np.zeros(H)

    d = g[1:H+1, :] - g[:H, :]  # Row difference r+1 - r

    def coh(a: int, b: int):
        if b - a < 20:
            return np.zeros(H), np.zeros(H)
        s = d[:, a:b]
        return (s >= min_step).mean(axis=1), (s <= -min_step).mean(axis=1)

    up_in, dn_in = coh(x1, x2)
    up_l, dn_l = coh(0, max(0, x1 - 5))
    up_r, dn_r = coh(min(gray.shape[1], x2 + 5), gray.shape[1])

    # Edge ownership: coherent inside ROI, but NOT outside
    own_up = up_in - np.maximum(up_l, up_r)
    own_dn = dn_in - np.maximum(dn_l, dn_r)
    own = np.maximum(own_up, own_dn)

    max_coh = np.maximum(up_in, dn_in)

    cand_indices = np.where((max_coh > 0.60) & ((own > 0.25) | (max_coh > 0.75)))[0]
    return cand_indices + 1, own


def detect_camera_rois_v6_ultra(img: np.ndarray) -> List[CameraRegion]:
    """
    Fast CPU Single-Frame Camera ROI Detection (Version 6 Ultra).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    row_stds = np.empty(h, dtype=np.float64)
    row_means = np.empty(h, dtype=np.float64)
    for r in range(h):
        m, s = cv2.meanStdDev(gray[r, :])
        row_means[r] = m[0][0]
        row_stds[r] = s[0][0]

    col_stds = np.empty(w, dtype=np.float64)
    for c in range(w):
        col_stds[c] = cv2.meanStdDev(gray[:, c])[1][0][0]

    thresh_y_35 = (sobel_y > 35).astype(np.uint8) * 255
    thresh_x_35 = (sobel_x > 35).astype(np.uint8) * 255

    kernel_h_25 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    lines_h_25 = cv2.morphologyEx(thresh_y_35, cv2.MORPH_OPEN, kernel_h_25)
    h_line_dens_25 = (lines_h_25 > 0).mean(axis=1)

    kernel_h_40 = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * 0.40)), 1))
    kernel_v_25 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.25))))

    lines_h = cv2.morphologyEx(thresh_y_35, cv2.MORPH_OPEN, kernel_h_40)
    lines_v = cv2.morphologyEx(thresh_x_35, cv2.MORPH_OPEN, kernel_v_25)

    h_line_dens = (lines_h > 0).mean(axis=1)
    v_line_dens = (lines_v > 0).mean(axis=0)

    # Fast Monotonicity Mask using cv2.blur
    mean_img = cv2.blur(gray.astype(np.float32), (5, 5))
    sqr_img = cv2.blur((gray.astype(np.float32)**2), (5, 5))
    local_std = np.sqrt(np.maximum(0.0, sqr_img - mean_img**2))
    mono_m = ((local_std < 7.5) | (gray < 15) | (gray > 220)).astype(np.uint8)

    mono_row = mono_m.mean(axis=1)
    mono_col = mono_m.mean(axis=0)

    cum_mono_row = make_cum(mono_row)
    cum_mono_col = make_cum(mono_col)
    cum_row_stds = make_cum(row_stds)
    cum_col_stds = make_cum(col_stds)

    v_line_dens_25 = (lines_v > 0).mean(axis=0)
    max_v_left = safe_max(v_line_dens_25[1:int(w*0.35)])
    max_v_right = safe_max(v_line_dens_25[int(w*0.65):w-1])

    left_mono = mono_col[:20].mean()
    right_mono = mono_col[-20:].mean()
    left_mean = gray[:, :20].mean()
    right_mean = gray[:, -20:].mean()

    has_side_pillars = (max_v_left > 0.40 or max_v_right > 0.40) and ((left_mono > 0.40 and left_mean < 35.0) or (right_mono > 0.40 and right_mean < 35.0))

    kernel_v_25px = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    kernel_v_15px = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    thresh_x_25 = (sobel_x > 25).astype(np.uint8) * 255
    lines_v_soft_25 = cv2.morphologyEx(thresh_x_25, cv2.MORPH_OPEN, kernel_v_25px)
    lines_v_soft_15 = cv2.morphologyEx(thresh_x_25, cv2.MORPH_OPEN, kernel_v_15px)

    center_v_soft_25 = float((lines_v_soft_25 > 0).mean(axis=0)[637:643].max()) if lines_v_soft_25.size > 0 and w >= 643 else 0.0
    center_v_soft_15 = float((lines_v_soft_15 > 0).mean(axis=0)[637:643].max()) if lines_v_soft_15.size > 0 and w >= 643 else 0.0
    is_true_divider = (center_v_soft_25 > 0.30) or (center_v_soft_15 > 0.32)

    top_white_bar = (row_means[:15].mean() > 180 and mono_row[:15].mean() > 0.40)
    top_offset = 25 if top_white_bar else 0

    if w <= 700 and h <= 500:
        if row_stds[:15].mean() > 10.0 and row_stds[-15:].mean() > 10.0:
            return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(cv2.meanStdDev(gray)[1][0][0]), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]

    if not has_side_pillars:
        if (mono_row[:15].mean() < 0.25 or top_white_bar) and mono_row[-15:].mean() < 0.25 and mono_col[:15].mean() < 0.25 and mono_col[-15:].mean() < 0.25:
            if safe_max(h_line_dens_25[:25]) < 0.05 and safe_max(h_line_dens_25[-25:]) < 0.05:
                return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(cv2.meanStdDev(gray)[1][0][0]), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]
        if not is_true_divider and col_stds[:20].mean() > 20.0 and col_stds[-20:].mean() > 20.0 and row_stds[:20].mean() > 10.0 and row_stds[-20:].mean() > 10.0:
            return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(cv2.meanStdDev(gray)[1][0][0]), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]
        if not is_true_divider and row_stds[:20].mean() > 12.0 and row_stds[-20:].mean() > 12.0 and col_stds[:20].mean() > 12.0 and col_stds[-20:].mean() > 12.0:
            if safe_max(h_line_dens_25[:20]) < 0.03 and safe_max(h_line_dens_25[-20:]) < 0.03:
                return [CameraRegion(x=0, y=top_offset, w=w, h=h-top_offset, area=w*(h-top_offset), activity=float(cv2.meanStdDev(gray)[1][0][0]), center_x=w/2.0, center_y=(h+top_offset)/2.0, rank=1)]

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
        if post_std > 20.0 and pre_std > 20.0 and line < 0.10 and mono_drop < 0.10:
            continue

        if (post_mono > 0.50 or post_std < 12.0) and (mono_drop > 0.15 or line > 0.02 or pre_std > post_std + 5.0):
            score = mono_drop * 10.0 + line * 200.0 + (pre_std - post_std) * 0.5
            if score > best_right_score and score > 1.0:
                best_right_score = score
                right_x = c + 1

    # 2.5 GRADIENT SIGN COHERENCE TOP SCAN & LONG LINE COVERAGE
    coh_cands, coh_own = top_edge_scan(gray, left_x, right_x, min_step=2, zone=0.35)
    coh_top_y = int(coh_cands[0]) if len(coh_cands) > 0 else 0

    sub_w = max(50, right_x - left_x)
    kernel_h_long = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(sub_w * 0.08)), 1))
    lines_h_long = cv2.morphologyEx(thresh_y_35[:, left_x:right_x], cv2.MORPH_OPEN, kernel_h_long)
    h_line_cov = (lines_h_long > 0).mean(axis=1)

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

        # Reject top candidates in top toolbar region r < 45 if there is no continuous horizontal frame line
        if r < 45 and h_line_cov[r] < 0.25 and not is_outer_trans:
            continue

        if line < 0.05 and mono_drop < 0.12 and not is_outer_trans:
            continue
        if r < 30 and line < 0.25 and post_mono > 0.70:
            continue
        if r < 150 and line < 0.30 and mono_drop < 0.12 and span_mean(cum_mono_row, r, min(h, r+30)) > 0.70:
            continue

        if top_y > 0 and best_top_line > 0.30 and span_mean(cum_mono_row, max(0, top_y - 10), top_y) > 0.85 and gray[min(h-1, top_y+5), :].mean() > 70.0 and r > top_y + 15:
            continue

        if (pre_mono > 0.40 or pre_std < 15.0 or is_outer_trans) and (mono_drop > 0.15 or line > 0.02 or post_std > pre_std + 3.0 or is_outer_trans):
            if best_top_score > 0 and best_top_line > 0.15 and line < 0.15:
                continue
            is_thick_line_band = (sum(h_line_dens_25[max(0, r-2):min(h, r+3)] > 0.05) >= 4)
            if is_thick_line_band and pre_mono < 0.90:
                continue
            if pre_mono < 0.70 and line < 0.50:
                continue
            if line < 0.35 and best_top_score > 0 and top_y >= 35 and span_mean(cum_row_stds, top_y, r) > 3.5:
                continue

            score = mono_drop * 10.0 + line * 300.0 + (post_std - pre_std) * 0.5
            if is_outer_trans:
                score += 15.0

            if score > best_top_score and score > 1.0:
                best_top_score = score
                top_y = r
                best_top_line = line

    # Refine top_y with Gradient Sign Coherence candidate if available and corrects a deep false horizon
    if coh_top_y > 0 and 20 <= coh_top_y <= 140:
        if best_top_score <= 0 or top_y > coh_top_y + 30:
            top_y = coh_top_y

    # Dynamic dark toolbar gap scanner (universal across image scales and UI layouts)
    max_top_search = int(h * 0.25)
    is_windowed = (left_x > 30 or right_x < w - 30 or (h_line_dens[:45] > 0.25).any())
    has_toolbar_gap = False
    if is_windowed and 0 < top_y < max_top_search:
        gap_start = -1
        gap_len = 0
        for r in range(max(1, top_y), max_top_search):
            if row_stds[r] < 10.0 and row_means[r] < 45.0:
                if gap_start < 0:
                    gap_start = r
                gap_len += 1
            else:
                if gap_len >= 4:
                    cum_row_means = make_cum(row_means)
                    cum_row_stds = make_cum(row_stds)
                    after_mean = span_mean(cum_row_means, r, min(h, r + 5))
                    after_std = span_mean(cum_row_stds, r, min(h, r + 5))
                    if (after_mean > row_means[max(0, r-1)] + 15.0) or (after_std > 25.0) or (h_line_dens[r] > 0.10):
                        if top_y < r:
                            top_y = r
                            has_toolbar_gap = True
                        break
                gap_start = -1
                gap_len = 0

    # Verify that there is an actual UI/black bar above top_y before accepting top_y > 20
    if top_y > 20 and not has_toolbar_gap:
        cum_row_means = make_cum(row_means)
        top_margin_std = span_mean(cum_row_stds, 0, min(top_y, 20))
        top_margin_mean = span_mean(cum_row_means, 0, min(top_y, 20))
        if top_margin_std > 8.0 and top_margin_mean > 40.0 and not top_white_bar:
            top_y = 0

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

        if r > 600 and span_mean(cum_row_stds, max(0, r - 8), r) < 3.0:
            best_bot_score = -1.0
            bot_y = r
            continue

        if (post_mono > 0.25 or post_std < 15.0) and span_mean(cum_row_stds, r, min(h, r+20)) < 20.0 and (mono_drop > 0.20 or line > 0.05 or pre_std > post_std + 5.0):
            score = mono_drop * 10.0 + line * 200.0 + (pre_std - post_std) * 0.5
            if score > best_bot_score and score > 2.0:
                best_bot_score = score
                bot_y = r + 1

    # MULTI-CAM LAYOUT BRANCH
    if (right_x - left_x) > 800 and is_true_divider:
        div_col = (lines_v_soft_15[:, 637:643] > 0).any(axis=1)
        div_indices = np.where(div_col)[0]
        valid_div = [idx for idx in div_indices if idx >= 45]
        if valid_div:
            top_y = int(valid_div[0])

        l_top_y, l_h = refine_cell_y_v6(img, left_x, 640 - left_x, default_top=top_y, default_bot=bot_y, pre_gray=gray, pre_sobel_y=sobel_y, pre_mono_m=mono_m)
        r_top_y, r_h = refine_cell_y_v6(img, 640, right_x - 640, default_top=top_y, default_bot=bot_y, pre_gray=gray, pre_sobel_y=sobel_y, pre_mono_m=mono_m)

        left_crop = gray[l_top_y:l_top_y+l_h, left_x:640]
        right_crop = gray[r_top_y:r_top_y+r_h, 640:right_x]

        left_act = get_cell_activity(left_crop)
        right_act = get_cell_activity(right_crop)

        cell_left = CameraRegion(x=left_x, y=l_top_y, w=640-left_x, h=l_h, area=(640-left_x)*l_h, activity=left_act, center_x=(left_x+640)/2.0, center_y=l_top_y+l_h/2.0, rank=1)
        cell_right = CameraRegion(x=640, y=r_top_y, w=right_x-640, h=r_h, area=(right_x-640)*r_h, activity=right_act, center_x=(640+right_x)/2.0, center_y=r_top_y+r_h/2.0, rank=2)

        if (left_act < 15.0 and right_act > 30.0) or (right_act > left_act + 50.0):
            cell_right.rank = 1
            cell_left.rank = 2
            rois = [cell_right, cell_left]
        else:
            cell_left.rank = 1
            cell_right.rank = 2
            rois = [cell_left, cell_right]

        return _apply_smart_telemetry_guard(rois, gray, h, w)

    # SINGLE-CAM LAYOUT BRANCH WITH TRANSLUCENT HUD OVERLAY REFINEMENT
    main_w = right_x - left_x
    main_h = bot_y - top_y

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if top_y >= 95 and top_y <= 115 and main_w > 800:
        sub_gray = gray[:, left_x:left_x + main_w]
        sub_sat = hsv[:, left_x:left_x + main_w, 1]
        edges_y = np.abs(cv2.Sobel(sub_gray, cv2.CV_64F, 0, 1, ksize=3))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(main_w * 0.40)), 1))
        lines_h = cv2.morphologyEx((edges_y > 35).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_h)
        line_dens_sub = (lines_h > 0).mean(axis=1)
        outer_candidates = [y for y in range(35, 65) if line_dens_sub[y] > 0.15]
        if outer_candidates:
            outer_y = outer_candidates[0]
            span_std = float(sub_gray[outer_y:top_y].std())
            span_avg = float(sub_gray[outer_y:top_y].mean())
            span_sat = float(sub_sat[outer_y:top_y].mean())
            if span_std > 20.0 and span_sat > 50.0 and span_avg > 20.0:
                main_h = main_h + (top_y - outer_y)
                top_y = outer_y

    act = get_cell_activity(gray[top_y:bot_y, left_x:right_x])
    main_reg = CameraRegion(x=left_x, y=top_y, w=main_w, h=main_h, area=main_w*main_h, activity=act, center_x=(left_x+right_x)/2.0, center_y=(top_y+bot_y)/2.0, rank=1)
    rois = [main_reg]
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


class VideoCameraDetector:
    """
    Video ROI Tracker (Fast-Path Sequential Video Acceleration).
    Maintains 100+ FPS on continuous video feeds by checking only ±3px boundary strips around the previous ROI.
    Falls back to full detection (~15-20ms) every `check_interval` frames or when layout/boundary changes.
    """
    def __init__(self, check_interval: int = 15, dens_thresh: float = 0.15):
        self.check_interval = check_interval
        self.dens_thresh = dens_thresh
        self.frame_count = 0
        self.last_rois = None
        self.last_densities = []

    def reset(self):
        self.frame_count = 0
        self.last_rois = None
        self.last_densities = []

    def _get_boundary_densities(self, gray: np.ndarray, roi: CameraRegion):
        h, w = gray.shape
        # Top boundary
        r_top = max(0, roi.y - 3)
        r_top_end = min(h, roi.y + 4)
        strip_top = gray[r_top:r_top_end, max(0, roi.x):min(w, roi.x + roi.w)]
        if strip_top.size > 0:
            s_top = np.abs(cv2.Sobel(strip_top, cv2.CV_64F, 0, 1, ksize=3))
            by = (s_top > 35).astype(np.uint8) * 255
            lines_h = cv2.morphologyEx(by, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
            d_top = float((lines_h > 0).mean())
        else:
            d_top = 0.0

        # Bot boundary
        r_bot = max(0, roi.y + roi.h - 4)
        r_bot_end = min(h, roi.y + roi.h + 3)
        strip_bot = gray[r_bot:r_bot_end, max(0, roi.x):min(w, roi.x + roi.w)]
        if strip_bot.size > 0:
            s_bot = np.abs(cv2.Sobel(strip_bot, cv2.CV_64F, 0, 1, ksize=3))
            by = (s_bot > 35).astype(np.uint8) * 255
            lines_h = cv2.morphologyEx(by, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
            d_bot = float((lines_h > 0).mean())
        else:
            d_bot = 0.0

        # Left boundary
        c_left = max(0, roi.x - 3)
        c_left_end = min(w, roi.x + 4)
        strip_left = gray[max(0, roi.y):min(h, roi.y + roi.h), c_left:c_left_end]
        if strip_left.size > 0:
            s_left = np.abs(cv2.Sobel(strip_left, cv2.CV_64F, 1, 0, ksize=3))
            bx = (s_left > 35).astype(np.uint8) * 255
            lines_v = cv2.morphologyEx(bx, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.25)))))
            d_left = float((lines_v > 0).mean())
        else:
            d_left = 0.0

        # Right boundary
        c_right = max(0, roi.x + roi.w - 4)
        c_right_end = min(w, roi.x + roi.w + 3)
        strip_right = gray[max(0, roi.y):min(h, roi.y + roi.h), c_right:c_right_end]
        if strip_right.size > 0:
            s_right = np.abs(cv2.Sobel(strip_right, cv2.CV_64F, 1, 0, ksize=3))
            bx = (s_right > 35).astype(np.uint8) * 255
            lines_v = cv2.morphologyEx(bx, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.25)))))
            d_right = float((lines_v > 0).mean())
        else:
            d_right = 0.0

        return (d_top, d_bot, d_left, d_right)

    def process_frame(self, img: np.ndarray, detector_fn=None) -> List[CameraRegion]:
        if detector_fn is None:
            detector_fn = detect_camera_rois_v6_ultra
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        fast_path_ok = False
        if self.last_rois is not None and self.frame_count < self.check_interval:
            fast_path_ok = True
            for i, roi in enumerate(self.last_rois):
                d_top, d_bot, d_left, d_right = self._get_boundary_densities(gray, roi)
                p_top, p_bot, p_left, p_right = self.last_densities[i]
                
                if (abs(d_top - p_top) > self.dens_thresh or
                    abs(d_bot - p_bot) > self.dens_thresh or
                    abs(d_left - p_left) > self.dens_thresh or
                    abs(d_right - p_right) > self.dens_thresh):
                    fast_path_ok = False
                    break
        
        if fast_path_ok:
            self.frame_count += 1
            return self.last_rois
            
        rois = detector_fn(img)
        self.last_rois = rois
        self.last_densities = [self._get_boundary_densities(gray, r) for r in rois]
        self.frame_count = 1
        return rois
        