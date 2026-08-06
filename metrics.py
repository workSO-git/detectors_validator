import numpy as np
import cv2

def compute_box_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    box format: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def compute_mask_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) of two binary masks.
    """
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def calculate_precision_recall(tp, fp, fn):
    """
    Calculate Precision and Recall.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
