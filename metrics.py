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

def calculate_f1_score(precision, recall):
    """
    Calculate F1 Score.
    """
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def get_mask_centroid(mask):
    """Calculate centroid (x, y) of a binary mask."""
    M = cv2.moments((mask > 0).astype(np.uint8))
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def get_box_centroid(box):
    """Calculate centroid (x, y) of a bounding box [x1, y1, x2, y2]."""
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    return (x_center, y_center)

def calculate_jitter(centroids):
    """
    Calculate average pixel displacement (jitter) between consecutive frames.
    centroids: list of (x, y) tuples or None
    """
    displacements = []
    for i in range(1, len(centroids)):
        c1 = centroids[i-1]
        c2 = centroids[i]
        if c1 is not None and c2 is not None:
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            displacements.append(dist)
    
    return np.mean(displacements) if displacements else 0.0

def calculate_flicker_rate(detections_presence):
    """
    Calculate flicker rate based on presence (True/False) per frame.
    Returns percentage of frames that are part of a flicker (on->off or off->on transition).
    """
    if len(detections_presence) < 2:
        return 0.0
    
    flips = 0
    for i in range(1, len(detections_presence)):
        if detections_presence[i] != detections_presence[i-1]:
            flips += 1
            
    return (flips / (len(detections_presence) - 1)) * 100.0
