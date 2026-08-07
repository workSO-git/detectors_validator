import os
import sys
import time
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
from evaluator import polygon_to_mask, box_to_absolute, _load_gt_from_file, _load_gt_from_json, _match_predictions
from metrics import compute_mask_iou, compute_box_iou, get_mask_centroid, get_box_centroid, calculate_jitter, calculate_flicker_rate, calculate_precision_recall, calculate_f1_score

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

TEMP_DIR = current_dir / "temp"
TEMP_DIR.mkdir(exist_ok=True)

from models.yolo_model import YoloModel

app = FastAPI(title="Horizon Segmentation Viewer")

# Model path (assuming the one we used previously)
MODEL_PATH = r"D:\work\seg_training\runs\drone_seg_n\weights\best.pt"

# Initialize model globally
model = None

@app.on_event("startup")
async def startup_event():
    global model
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
            
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YoloModel(MODEL_PATH, task='seg')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

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
        results = model.model.predict(img, conf=0.25, verbose=False, retina_masks=True)
        res = results[0]

        img_all   = res.plot(boxes=True,  masks=True)
        img_masks = res.plot(boxes=False, masks=True)
        img_boxes = res.plot(boxes=True,  masks=False)

        inf_time    = sum(res.speed.values()) if hasattr(res, 'speed') else 0
        num_objects = len(res.masks.data) if (hasattr(res, 'masks') and res.masks is not None) else 0

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
                
            gt_list = gt_masks if model.task == 'seg' else gt_boxes

            pred_list = []
            if model.task == 'seg' and res.masks is not None:
                for mask_t in res.masks.data:
                    m = mask_t.cpu().numpy()
                    pred_list.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
            elif model.task == 'det' and res.boxes is not None:
                for box in res.boxes.xyxy.cpu().numpy():
                    pred_list.append(box.tolist())

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
        "paused": False,
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
                    elif cmd == "set_filter":
                        state["filter"] = data["filter"]
                    elif cmd == "pause":
                        state["paused"] = data["state"]
                    elif cmd == "set_labels_dir":
                        state["labels_dir"] = data["path"] if data["path"] else None
                        buffer_ious.clear()
                        buffer_centroids.clear()
                        buffer_detections.clear()
        except Exception:
            pass

    recv_task = asyncio.create_task(receive_commands())

    try:
        while True:
            if recv_task.done():
                break

            force_read = False
            if state["seek"] is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state["seek"])
                state["seek"] = None
                force_read = True

            if state["paused"] and not force_read:
                await asyncio.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"type": "done"})
                if state["paused"]:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # inference
            start_time = time.time()
            results = model.model.predict(frame, conf=0.25, verbose=False, retina_masks=True)
            res = results[0]
            
            # Simulated model state (if the actual model has `warming_up` attribute)
            warming_up = getattr(model, 'warming_up', False)
            metrics_payload = None
            
            # --- Live Metrics Logic ---
            if state["labels_dir"]:
                labels_path = Path(state["labels_dir"])
                stem = f"{int(frame_idx):04d}"
                h, w = frame.shape[:2]
                
                if warming_up:
                    metrics_payload = {"warming_up": True, "has_data": False}
                else:
                    gt_masks, gt_boxes = [], []
                    has_gt_annotation = False
                    
                    if labels_path.suffix.lower() == '.json':
                        gt_masks, gt_boxes, has_gt_annotation = _load_gt_from_json(labels_path, stem, w, h, model.task)
                    else:
                        lbl_file = resolve_label_file(labels_path, stem)
                        if lbl_file and lbl_file.exists():
                            gt_masks, gt_boxes = _load_gt_from_file(lbl_file, model.task, w, h)
                            has_gt_annotation = True
                                
                    gt_list = gt_masks if model.task == 'seg' else gt_boxes
                    
                    if has_gt_annotation:
                        # pred
                        pred_masks = []
                        if model.task == 'seg' and res.masks is not None:
                            for i, mask_tensor in enumerate(res.masks.data):
                                mask = mask_tensor.cpu().numpy()
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                                pred_masks.append(mask)
                                
                        has_detection = len(pred_masks) > 0
                        buffer_detections.append(has_detection)
                        
                        if has_detection:
                            buffer_centroids.append(get_mask_centroid(pred_masks[0]))
                        else:
                            buffer_centroids.append(None)
                            
                        # Frame IoU
                        frame_iou = 0.0
                        matched_gt = set()
                        for p_mask in pred_masks:
                            best_iou, best_gt_idx = 0, -1
                            for i, g_mask in enumerate(gt_masks):
                                if i in matched_gt: continue
                                iou = compute_mask_iou(p_mask, g_mask)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = i
                            if best_iou >= 0.5:
                                matched_gt.add(best_gt_idx)
                                frame_iou = max(frame_iou, best_iou)
                                
                        if len(gt_masks) > 0:
                            buffer_ious.append(frame_iou)
                        else:
                            # if 0 GT objects and model predicted something, IoU is 0. If model predicted nothing, IoU is 1.
                            buffer_ious.append(1.0 if len(pred_masks) == 0 else 0.0)
                        
                        metrics_payload = {
                            "warming_up": False,
                            "has_data": True,
                            "iou_avg": float(np.mean(buffer_ious)),
                            "iou_min": float(np.min(buffer_ious)),
                            "jitter": float(calculate_jitter(list(buffer_centroids))),
                            "flicker": float(calculate_flicker_rate(list(buffer_detections)))
                        }
                    else:
                        metrics_payload = {"warming_up": False, "has_data": False}
            # --------------------------
            
            b_flag = state["filter"] in ["all", "boxes"]
            m_flag = state["filter"] in ["all", "masks"]
            plotted_img = res.plot(boxes=b_flag, masks=m_flag)
            
            inf_time = sum(res.speed.values()) if hasattr(res, 'speed') else 0
            
            _, buffer = cv2.imencode('.jpg', plotted_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65]) 
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            _, buffer_orig = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
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
            
            await asyncio.sleep(0.001) # Yield to event loop
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
