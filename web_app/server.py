import os
import sys
import time
import math
import base64
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import asyncio
import collections


def _safe_float(v):
    """Convert to float, returning None for NaN/Inf (which break JSON serialization)."""
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) or math.isinf(f) else f


def _smart_iou(p, g, shape):
    """Compute IoU between prediction and GT, handling mask/box mixed types."""
    from metrics import compute_mask_iou, compute_box_iou
    is_p_mask = isinstance(p, np.ndarray)
    is_g_mask = isinstance(g, np.ndarray)
    if is_p_mask and is_g_mask:
        return compute_mask_iou(p, g)
    elif not is_p_mask and not is_g_mask:
        return compute_box_iou(p, g)
    elif is_p_mask and not is_g_mask:
        g_mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.rectangle(g_mask, (g[0], g[1]), (g[2], g[3]), 1, -1)
        return compute_mask_iou(p, g_mask)
    else:
        p_mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.rectangle(p_mask, (p[0], p[1]), (p[2], p[3]), 1, -1)
        return compute_mask_iou(p_mask, g)

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from evaluator import polygon_to_mask, box_to_absolute, _load_gt_from_file, _load_gt_from_json, _match_predictions
from metrics import compute_mask_iou, compute_box_iou, get_mask_centroid, get_box_centroid, calculate_jitter, calculate_flicker_rate, calculate_precision_recall, calculate_f1_score

TEMP_DIR = current_dir / "temp"
TEMP_DIR.mkdir(exist_ok=True)

from main import get_model_adapter

app = FastAPI(title="Horizon Segmentation Viewer")

AVAILABLE_MODELS = [
    {
        "id": "vidi_yolo",
        "name": "YOLO Interface Detector (VIDI)",
        "model_type": "interface",
        "model_path": {
            "detector_module": "detectors.yolo_interface_detector",
            "detector_class": "YOLOInterfaceDetector",
            "model_path": r"C:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt"
        },
        "task": "det"
    },
    {
        "id": "vidi_geometric",
        "name": "Geometric Detector (VIDI)",
        "model_type": "interface",
        "model_path": "detectors.geometric_detector:GeometricDetector",
        "task": "det"
    },
    {
        "id": "gmm_detector",
        "name": "GMM Detector (VIDI)",
        "model_type": "interface",
        "model_path": {"detector_module": "detectors.gmm_detector", "detector_class": "GMMDetector"},
        "task": "ignore"
    },
    {
        "id": "ema_detector",
        "name": "EMA Detector (test_gmm)",
        "model_type": "interface",
        "model_path": {"detector_module": "detectors.ema_detector", "detector_class": "EMADetector"},
        "task": "ignore"
    },
    {
        "id": "vidi_mask",
        "name": "Mask Detector (VIDI)",
        "model_type": "interface",
        "model_path": "detectors.mask_detector:MaskDetector",
        "task": "det"
    },
    {
        "id": "ignore_mask",
        "name": "Ignore Adapter (Masks)",
        "model_type": "ignore",
        "model_path": "None",
        "task": "seg"
    },
    {
        "id": "yolo_det",
        "name": "YOLO Object Detection (Default)",
        "model_type": "yolo",
        "model_path": r"c:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt",
        "task": "det"
    },
    {
        "id": "yolo_seg",
        "name": "YOLO Segmentation",
        "model_type": "yolo",
        "model_path": r"C:\Users\Sasha\projects\CV\models_extracted\models\best.pt",
        "task": "seg"
    }
]

# Initialize model globally
model = None
current_model_id = None

@app.on_event("startup")
async def startup_event():
    global model, current_model_id
    print("Cleaning up temporary files...")
    import shutil
    for item in TEMP_DIR.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"Failed to delete {item}: {e}")
            
    # Load default model
    default_cfg = AVAILABLE_MODELS[0]
    current_model_id = default_cfg["id"]
    print(f"Loading default model {default_cfg['name']}...")
    try:
        model = get_model_adapter(default_cfg['model_type'], default_cfg['model_path'], default_cfg['task'])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading default model: {e}")

def perform_inference_and_draw(model_adapter, img):
    """
    Unified inference and drawing function for both YOLO and generic adapters.
    Returns: (img_all, img_masks, img_boxes, inf_time_ms, num_objects, raw_results)
    """
    import time
    if hasattr(model_adapter, 'model') and hasattr(model_adapter.model, 'predict'):
        # YOLO specific fast-path with built-in plotting
        results = model_adapter.model.predict(img, conf=0.25, verbose=False, retina_masks=True)
        res = results[0]
        img_all   = res.plot(boxes=True,  masks=True)
        img_masks = res.plot(boxes=False, masks=True)
        img_boxes = res.plot(boxes=True,  masks=False)
        inf_time  = sum(res.speed.values()) if hasattr(res, 'speed') else 0
        num_objects = len(res.masks.data) if getattr(res, 'masks', None) else len(res.boxes)
        return img_all, img_masks, img_boxes, inf_time, num_objects, res
    else:
        # Generic adapter path
        t0 = time.time()
        preds = model_adapter.predict(img)
        inf_time = (time.time() - t0) * 1000
        
        boxes = preds.get('boxes', [])
        masks = preds.get('masks', [])
        num_objects = max(len(boxes), len(masks))
        
        img_all = img.copy()
        img_masks = img.copy()
        img_boxes = img.copy()
        
        # Draw masks
        for m in masks:
            if isinstance(m, np.ndarray):
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[m > 0] = (0, 0, 255) # Red mask
                cv2.addWeighted(colored_mask, 0.5, img_all, 1.0, 0, img_all)
                cv2.addWeighted(colored_mask, 0.5, img_masks, 1.0, 0, img_masks)
                
        # Draw boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img_all, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label
            cv2.putText(img_all, "det", (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img_boxes, "det", (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Create a mock result object that mimics YOLO's res.masks and res.boxes for websocket metrics compatibility
        class MockRes:
            def __init__(self, b, m):
                class MockBoxes:
                    def __init__(self, b): self.data = b
                class MockMasks:
                    def __init__(self, m):
                        import torch
                        # Websocket expects res.masks.data to be an iterable of tensors with .cpu().numpy()
                        self.data = [torch.tensor(mask) for mask in m]
                self.boxes = MockBoxes(b)
                self.masks = MockMasks(m) if m else None
        
        return img_all, img_masks, img_boxes, inf_time, num_objects, MockRes(boxes, masks)

@app.get("/api/models")
async def get_models():
    return {"models": AVAILABLE_MODELS, "current": current_model_id}

@app.post("/api/set_model")
async def set_model(model_id: str = Form(...)):
    global model, current_model_id
    cfg = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if not cfg:
        return JSONResponse(status_code=404, content={"error": "Model not found"})
    try:
        print(f"Switching model to {cfg['name']}...")
        model = get_model_adapter(cfg['model_type'], cfg['model_path'], cfg['task'])
        current_model_id = model_id
        return {"success": True, "model": cfg}
    except Exception as e:
        print(f"Error switching model: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/pick_folder")
async def pick_folder():
    """Open a native OS folder picker dialog and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog

    def _open_dialog():
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Виберіть теку")
        root.destroy()
        return folder or ""

    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, _open_dialog)
    return {"path": path}

@app.get("/api/pick_file")
async def pick_file():
    """Open a native OS file picker dialog for YAML files."""
    import tkinter as tk
    from tkinter import filedialog

    def _open_dialog():
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Виберіть dataset.yaml або .json",
            filetypes=[("YAML / JSON", "*.yaml *.yml *.json"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path or ""

    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, _open_dialog)
    return {"path": path}

def resolve_label_file(labels_path: Path, stem: str) -> Path:
    """Resolve the .txt label file for a given image/frame stem, supporting dataset.yaml paths."""
    if labels_path.is_file() and labels_path.suffix in ['.yaml', '.yml']:
        base_dir = labels_path.parent
        # Attempt to find the label file in standard YOLO structure
        for split in ['val', 'train', 'test', '']:
            candidate = base_dir / 'labels' / split / f"{stem}.txt"
            if candidate.exists():
                return candidate
        # Fallback to recursive search if not found
        for candidate in base_dir.rglob(f"{stem}.txt"):
            if 'labels' in candidate.parts:
                return candidate
        return None
    else:
        # It's a directory
        return labels_path / f"{stem}.txt"

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    labels_dir: str = Form(default=""),
    no_images: str = Form(default="false")
):
    skip_images = no_images.lower() in ("true", "1", "yes")
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        h, w = img.shape[:2]

        # Run inference
        img_all, img_masks, img_boxes, inf_time, num_objects, res = perform_inference_and_draw(model, img)

        def encode(im):
            _, buf = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        if skip_images:
            response = {
                "success":  True,
                "image":    None,
                "image_masks": None,
                "image_boxes": None,
                "time_ms":  round(inf_time, 2),
                "polygons": num_objects,
                "filename": file.filename,
                "metrics":  None
            }
        else:
            response = {
                "success":     True,
                "image":       encode(img_all),
                "image_masks": encode(img_masks),
                "image_boxes": encode(img_boxes),
                "time_ms":     round(inf_time, 2),
                "polygons":    num_objects,
                "filename":    file.filename,
                "metrics":     None
            }

        # If labels_dir is provided, evaluate vs GT
        if labels_dir.strip():
            stem        = Path(file.filename).stem
            labels_path = Path(labels_dir.strip())
            
            gt_masks, gt_boxes = [], []
            lbl_file = None
            has_gt_annotation = False
            
            if labels_path.suffix.lower() == '.json':
                gt_masks, gt_boxes, has_gt_annotation = _load_gt_from_json(labels_path, file.filename, w, h, model.task)
            else:
                lbl_file = resolve_label_file(labels_path, stem)
                if lbl_file and lbl_file.exists():
                    gt_masks, gt_boxes = _load_gt_from_file(lbl_file, model.task, w, h)
                    has_gt_annotation = True

            # Create an image overlay for Ground Truth if annotations exist
            if has_gt_annotation:
                img_gt = img.copy()
                if gt_masks:
                    # Draw GT masks with proper alpha blending so it darkens bright skies
                    alpha = 0.7
                    color = np.array([0, 0, 255], dtype=np.uint8) # Solid Red in BGR
                    for mask in gt_masks:
                        mask_bool = mask > 0
                        img_gt[mask_bool] = (img_gt[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)
                if gt_boxes:
                    # Draw GT boxes in Solid Red
                    for box in gt_boxes:
                        cv2.rectangle(img_gt, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                
                response["image_gt"] = encode(img_gt)
                
            gt_list = gt_masks if model.task == 'seg' else gt_boxes

            pred_list = []
            if model.task == 'seg' and res.masks is not None:
                for mask_tensor in res.masks.data:
                    if hasattr(mask_tensor, 'cpu'):
                        mask = mask_tensor.cpu().numpy()
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask = mask_tensor.numpy() if hasattr(mask_tensor, 'numpy') else mask_tensor
                        mask = cv2.resize(np.array(mask, dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_list.append(mask)
            elif model.task in ('det', 'ignore') and res.boxes is not None:
                for box_tensor in res.boxes.data:
                    if hasattr(box_tensor, 'cpu'):
                        b = box_tensor.cpu().numpy()
                    else:
                        b = box_tensor
                    pred_list.append([float(v) for v in b[:4]])

            iou_fn = compute_mask_iou if model.task == 'seg' else compute_box_iou
            if has_gt_annotation:
                tp, fp, fn, ious = _match_predictions(pred_list, gt_list, iou_fn, 0.5)
                precision, recall = calculate_precision_recall(tp, fp, fn)
                f1 = calculate_f1_score(precision, recall)
                
                # Image-level IoU logic
                if len(gt_list) == 0 and len(pred_list) == 0:
                    image_iou = 1.0  # True Negative
                elif ious:
                    image_iou = float(np.mean(ious))
                else:
                    image_iou = 0.0  # False Positive or False Negative without any matches
                    
                response["metrics"] = {
                    "has_gt":    True,
                    "iou":       round(image_iou, 4),
                    "precision": round(precision, 4),
                    "recall":    round(recall, 4),
                    "f1":        round(f1, 4),
                    "objects":   num_objects,
                    "gt_count":  len(gt_list)
                }
            else:
                response["metrics"] = {"has_gt": False, "reason": "GT файл розмітки не знайдено"}

        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/upload_video")
async def upload_video(request: Request, filename: str):
    try:
        file_path = TEMP_DIR / f"{uuid.uuid4()}_{filename}"
        with open(file_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        return {"success": True, "path": str(file_path)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket, path: str):
    await websocket.accept()
    if model is None:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close()
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        await websocket.send_json({"error": "Cannot open video file"})
        await websocket.close()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    await websocket.send_json({"type": "init", "total_frames": total_frames})

    state = {
        "seek": None,
        "filter": "all",
        "paused": True,   # Start paused — client sends 'pause: false' to begin playback
        "labels_dir": None
    }
    
    # Buffers for rolling window (50 frames)
    buffer_ious = collections.deque(maxlen=50)
    buffer_centroids = collections.deque(maxlen=50)
    buffer_detections = collections.deque(maxlen=50)

    async def receive_commands():
        try:
            while True:
                data = await websocket.receive_json()
                if "command" in data:
                    cmd = data["command"]
                    if cmd == "seek":
                        state["seek"] = data["frame"]
                        buffer_ious.clear()
                        buffer_centroids.clear()
                        buffer_detections.clear()
                        # Reset GMM Detector on seek so it doesn't get corrupted by disjointed frames
                        if hasattr(model, '_detector') and hasattr(model._detector, 'frames_processed'):
                            model._detector.frames_processed = 0
                            if hasattr(model._detector, 'back_sub'):
                                model._detector.back_sub = cv2.createBackgroundSubtractorMOG2(
                                    history=model._detector.back_sub.getHistory(),
                                    varThreshold=model._detector.back_sub.getVarThreshold(),
                                    detectShadows=model._detector.back_sub.getDetectShadows()
                                )
                    elif cmd == "set_filter":
                        state["filter"] = data["filter"]
                    elif cmd == "pause":
                        state["paused"] = data["state"]
                    elif cmd == "set_labels_dir":
                        path = data.get("path", "")
                        if path:
                            path = path.strip('\'"')
                        state["labels_dir"] = path if path else None
                        print(f"DEBUG: set_labels_dir received! path={state['labels_dir']}")
                        buffer_ious.clear()
                        buffer_centroids.clear()
                        buffer_detections.clear()
        except Exception as e:
            print(f"receive_commands error: {e}")

    # IgnoreAdapter dynamic training: train on the current video before streaming
    if hasattr(model, 'train') and getattr(model, '__class__', None).__name__ == 'IgnoreAdapter':
        print(f"[Server] Dynamic training for IgnoreAdapter on video: {path}")
        model.train([path], total_sample_count=150)

    # Ensure GMM Detector (and others) knows it's streaming a video
    if hasattr(model, '_detector'):
        if hasattr(model._detector, 'set_video_mode'):
            model._detector.set_video_mode(True)
        # Reset background learning if it's a new video
        if hasattr(model._detector, 'frames_processed'):
            model._detector.frames_processed = 0
            if hasattr(model._detector, 'back_sub'):
                # Note: cv2 is globally imported at the top of server.py
                model._detector.back_sub = cv2.createBackgroundSubtractorMOG2(
                    history=model._detector.back_sub.getHistory(),
                    varThreshold=model._detector.back_sub.getVarThreshold(),
                    detectShadows=model._detector.back_sub.getDetectShadows()
                )

    recv_task = asyncio.create_task(receive_commands())

    try:
        while True:
            # recv_task is intentionally NOT checked here.
            # If receive_commands() fails (e.g. client sent bad JSON), we keep streaming.
            # The loop will exit naturally when cap.read() returns False or WebSocket closes.

            loop_start_time = time.time()
            force_read = False
            if state["seek"] is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state["seek"])
                state["seek"] = None
                force_read = True

            if state["paused"] and not force_read:
                await asyncio.sleep(0.05)
                continue
            
            # If we just seeked while paused, read + send exactly one frame, then pause again
            re_pause_after = state["paused"] and force_read

            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"type": "done"})
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # inference
            start_time = time.time()
            img_all, img_masks, img_boxes, inf_time, num_objects, res = perform_inference_and_draw(model, frame)
            
            h, w = frame.shape[:2]
            img_orig = frame.copy()
            
            # Simulated model state (if the actual model has `warming_up` attribute)
            warming_up = getattr(model, 'warming_up', False)
            metrics_payload = None
            
            # --- Live Metrics Logic ---
            if state["labels_dir"]:
                labels_path = Path(state["labels_dir"])
                stem = f"{int(frame_idx):04d}"
                
                if warming_up:
                    metrics_payload = {"warming_up": True, "has_data": False}
                else:
                    gt_masks, gt_boxes = [], []
                    has_gt_annotation = False
                    
                    if labels_path.suffix.lower() == '.json':
                        gt_masks, gt_boxes, has_gt_annotation = _load_gt_from_json(labels_path, stem, w, h, model.task)
                        print(f"DEBUG JSON: stem={stem}, task={model.task}, found={has_gt_annotation}, boxes={len(gt_boxes)}")
                    else:
                        lbl_file = resolve_label_file(labels_path, stem)
                        if lbl_file and lbl_file.exists():
                            gt_masks, gt_boxes = _load_gt_from_file(lbl_file, model.task, w, h)
                            has_gt_annotation = True
                            print(f"DEBUG TXT: found={has_gt_annotation}")
                        else:
                            print(f"DEBUG TXT: NOT FOUND {lbl_file}")
                                
                    gt_list = gt_masks if model.task == 'seg' else gt_boxes
                    
                    if has_gt_annotation:
                        # pred
                        pred_items = []
                        if model.task == 'seg' and res.masks is not None:
                            for mask_tensor in res.masks.data:
                                if hasattr(mask_tensor, 'cpu'):
                                    mask = mask_tensor.cpu().numpy()
                                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                else:
                                    mask = mask_tensor.numpy() if hasattr(mask_tensor, 'numpy') else mask_tensor
                                    mask = cv2.resize(np.array(mask, dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                                pred_items.append(mask)
                        elif model.task in ('det', 'ignore'):
                            if res.boxes is not None and getattr(res.boxes, 'data', []) is not None and len(res.boxes.data) > 0:
                                for box_tensor in res.boxes.data:
                                    if hasattr(box_tensor, 'cpu'):
                                        b = box_tensor.cpu().numpy()
                                    else:
                                        b = box_tensor
                                    pred_items.append([int(v) for v in b[:4]])
                            elif res.masks is not None and getattr(res.masks, 'data', []) is not None and len(res.masks.data) > 0:
                                # The model predicted masks, but the task is 'ignore'.
                                # Append the masks directly. We will handle Mask vs Box IoU smartly.
                                for mask_tensor in res.masks.data:
                                    if hasattr(mask_tensor, 'cpu'):
                                        m = mask_tensor.cpu().numpy()
                                    else:
                                        m = mask_tensor.numpy() if hasattr(mask_tensor, 'numpy') else mask_tensor
                                    
                                    if m.ndim > 2: m = m.squeeze()
                                    binary_mask = (m > 0).astype(np.uint8)
                                    pred_items.append(cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST))
                                
                        has_detection = len(pred_items) > 0
                        buffer_detections.append(has_detection)
                        
                        if has_detection:
                            if isinstance(pred_items[0], np.ndarray):
                                buffer_centroids.append(get_mask_centroid(pred_items[0]))
                            else:
                                buffer_centroids.append(get_box_centroid(pred_items[0]))
                        else:
                            buffer_centroids.append(None)
                            
                        # Frame IoU
                        frame_iou = 0.0
                        matched_gt = set()
                        
                        if model.task == 'ignore':
                            # For ignore zones, we merge all GT and Pred items into single masks to compute global IoU
                            g_mask_full = np.zeros((h, w), dtype=np.uint8)
                            for g in gt_list:
                                if isinstance(g, np.ndarray):
                                    g_mask_full = np.logical_or(g_mask_full, g)
                                else:
                                    cv2.rectangle(g_mask_full, (g[0], g[1]), (g[2], g[3]), 1, -1)
                            
                            p_mask_full = np.zeros((h, w), dtype=np.uint8)
                            for p in pred_items:
                                if isinstance(p, np.ndarray):
                                    p_mask_full = np.logical_or(p_mask_full, p)
                                else:
                                    cv2.rectangle(p_mask_full, (p[0], p[1]), (p[2], p[3]), 1, -1)
                            
                            frame_iou = compute_mask_iou(p_mask_full.astype(np.uint8), g_mask_full.astype(np.uint8))
                        else:
                            for p_item in pred_items:
                                best_iou, best_gt_idx = 0, -1
                                for i, g_item in enumerate(gt_list):
                                    if i in matched_gt: continue
                                    iou = _smart_iou(p_item, g_item, frame.shape)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_gt_idx = i
                                        
                                if best_iou > 0.1:  # Threshold
                                    matched_gt.add(best_gt_idx)
                                    
                                frame_iou += best_iou
                                
                            if len(pred_items) > 0:
                                frame_iou /= len(pred_items)
                        
                        if len(gt_list) > 0:
                            buffer_ious.append(frame_iou)
                        else:
                            # if 0 GT objects and model predicted something, IoU is 0. If model predicted nothing, IoU is 1.
                            buffer_ious.append(1.0 if len(pred_items) == 0 else 0.0)
                        
                        metrics_payload = {
                            "warming_up": False,
                            "has_data": True,
                            "iou_cur": _safe_float(frame_iou),
                            "iou_avg": _safe_float(np.mean(buffer_ious)),
                            "iou_min": _safe_float(np.min(buffer_ious)),
                            "jitter": _safe_float(calculate_jitter(list(buffer_centroids))),
                            "flicker": _safe_float(calculate_flicker_rate(list(buffer_detections)))
                        }
                        
                        # Draw GT on img_orig so the user can see what is being evaluated against
                        for g in gt_list:
                            if isinstance(g, np.ndarray):
                                colored_mask = np.zeros_like(img_orig, dtype=np.uint8)
                                colored_mask[g > 0] = (255, 0, 0) # Blue for GT
                                cv2.addWeighted(colored_mask, 0.5, img_orig, 1.0, 0, img_orig)
                            else:
                                cv2.rectangle(img_orig, (g[0], g[1]), (g[2], g[3]), (255, 0, 0), 2)
                                cv2.putText(img_orig, "GT", (g[0], max(g[1]-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                
                    else:
                        metrics_payload = {"warming_up": False, "has_data": False}
            # --------------------------
            
            b_flag = state["filter"] in ["all", "boxes"]
            m_flag = state["filter"] in ["all", "masks"]
            
            if b_flag and m_flag:
                plotted_img = img_all
            elif b_flag:
                plotted_img = img_boxes
            elif m_flag:
                plotted_img = img_masks
            else:
                plotted_img = frame.copy()
                
            # Scale down for web streaming to prevent browser UI lag and memory leaks
            stream_h = 480
            stream_w = int(w * (stream_h / h))
            stream_img = cv2.resize(plotted_img, (stream_w, stream_h))
            stream_orig = cv2.resize(img_orig, (stream_w, stream_h))
            
            _, buffer = cv2.imencode('.jpg', stream_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65]) 
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            _, buffer_orig = cv2.imencode('.jpg', stream_orig, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            img_orig_base64 = base64.b64encode(buffer_orig).decode('utf-8')
            
            fps = 1.0 / (time.time() - start_time + 0.001)

            await websocket.send_json({
                "type": "frame",
                "image": f"data:image/jpeg;base64,{img_base64}",
                "image_orig": f"data:image/jpeg;base64,{img_orig_base64}",
                "fps": round(fps, 1),
                "time_ms": round(inf_time, 2),
                "frame_idx": frame_idx,
                "metrics": metrics_payload
            })
            
            if re_pause_after:
                state["paused"] = True

            elapsed = time.time() - loop_start_time
            sleep_amount = getattr(model, 'frame_delay', 0.033) - elapsed
            if sleep_amount > 0:
                await asyncio.sleep(sleep_amount)
            else:
                await asyncio.sleep(0.001)
    except WebSocketDisconnect:
        pass  # Normal client disconnect — no error needed
    except Exception as e:
        print(f"WS Error: {e}")
    finally:
        recv_task.cancel()
        cap.release()

# Mount the static directory for the frontend
app.mount("/", StaticFiles(directory=str(current_dir / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
