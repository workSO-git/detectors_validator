# COMMANDS — Команди для запуску оцінки моделей

Всі команди запускаються з кореневої папки проекту.

> **Перед запуском:** скопіюйте `.env.example` → `.env` і заповніть шляхи до ваг.
> Замініть `<MODELS_DIR>` та `<DATA_DIR>` на реальні шляхи у вашій системі.

---

## 🌐 Веб-інтерфейс

```powershell
python -m uvicorn web_app.server:app --reload --port 8000
```
Відкрити у браузері: http://localhost:8000

---

## 🎬 Тестування на ВІДЕО (`--mode video_eval`)

### YOLO Object Detection (det)
```powershell
python main.py --mode video_eval `
  --model-type yolo `
  --model "<MODELS_DIR>\yolo_det\weights\best.pt" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --labels "<DATA_DIR>\videos\my_video_gt.json" `
  --task det
```

### YOLO Segmentation (seg)
```powershell
python main.py --mode video_eval `
  --model-type yolo `
  --model "<MODELS_DIR>\yolo_seg\weights\best.pt" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --labels "<DATA_DIR>\videos\my_video_gt.json" `
  --task seg
```

### EMA Detector (Interface / VIDI)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.ema_detector', 'detector_class': 'EMADetector'}" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --labels "<DATA_DIR>\videos\my_video_gt.json" `
  --task det
```

### GMM Detector (Interface / VIDI)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.gmm_detector', 'detector_class': 'GMMDetector'}" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --labels "<DATA_DIR>\videos\my_video_gt.json" `
  --task det
```

### Geometric Detector (Interface / VIDI)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.geometric_detector', 'detector_class': 'GeometricDetector'}" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --labels "<DATA_DIR>\videos\my_video_gt.json" `
  --task det
```

### YOLO через VIDI-обгортку (YOLOInterfaceDetector)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.yolo_interface_detector', 'detector_class': 'YOLOInterfaceDetector', 'model_path': '<MODELS_DIR>\\yolo_det\\weights\\best.pt'}" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --labels "<DATA_DIR>\videos\my_video_gt.json" `
  --task det
```

### IgnoreAdapter (навчається на відео перед оцінкою)
```powershell
python main.py --mode video_eval `
  --model-type ignore `
  --model none `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --labels "<DATA_DIR>\videos\my_video_gt.json" `
  --task ignore
```

### Depth-Anything V2 (auto-download з HuggingFace)
```powershell
python main.py --mode video_eval `
  --model-type depth `
  --model none `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --task seg `
  --depth-threshold 0.5
```

### DINOv2 MLP Head
```powershell
python main.py --mode video_eval `
  --model-type dinov2 `
  --model "<MODELS_DIR>\best_dinov2_mlp_model.pth" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --task seg
```

### SMP ResNet
```powershell
python main.py --mode video_eval `
  --model-type reznet `
  --model "<MODELS_DIR>\best_sky_model.pth" `
  --source "<DATA_DIR>\videos\my_video.mp4" `
  --task seg
```

---

## 📂 Тестування на ДАТАСЕТАХ (`--mode dataset`)

### YOLO Detection — датасет у форматі YOLO
```powershell
python main.py --mode dataset `
  --model-type yolo `
  --model "<MODELS_DIR>\yolo_det\weights\best.pt" `
  --source "<DATA_DIR>\det_dataset" `
  --task det `
  --split val
```

### YOLO Segmentation
```powershell
python main.py --mode dataset `
  --model-type yolo `
  --model "<MODELS_DIR>\yolo_seg\weights\best.pt" `
  --source "<DATA_DIR>\seg_dataset" `
  --task seg `
  --split val
```

### SMP ResNet — датасет images+masks
```powershell
python main.py --mode mask_dataset `
  --model-type reznet `
  --model "<MODELS_DIR>\best_sky_model.pth" `
  --source "<DATA_DIR>\sky_dataset" `
  --task seg
```

---

## 🖼️ Тестування одного ЗОБРАЖЕННЯ (`--mode single`)

### YOLO
```powershell
python main.py --mode single `
  --model-type yolo `
  --model "<MODELS_DIR>\yolo_det\weights\best.pt" `
  --source "<DATA_DIR>\images\frame.jpg" `
  --task det `
  --save-dir results\
```

### DINOv2
```powershell
python main.py --mode single `
  --model-type dinov2 `
  --model "<MODELS_DIR>\best_dinov2_mlp_model.pth" `
  --source "<DATA_DIR>\images\frame.jpg" `
  --task seg `
  --save-dir results\
```

### Depth-Anything V2
```powershell
python main.py --mode single `
  --model-type depth `
  --model none `
  --source "<DATA_DIR>\images\frame.jpg" `
  --task seg `
  --depth-threshold 0.4 `
  --save-dir results\
```

---

## 🔧 Параметри `main.py`

| Параметр | Значення | Опис |
|----------|----------|------|
| `--mode` | `dataset` / `mask_dataset` / `single` / `video_eval` | Режим оцінки |
| `--model-type` | `yolo` / `reznet` / `dinov2` / `depth` / `interface` / `ignore` / `horizon` / `tracker` | Тип адаптера |
| `--model` | шлях або dict-рядок | Шлях до `.pt`/`.pth`/`.onnx` або конфіг детектора |
| `--source` | шлях | Папка датасету, відеофайл або зображення |
| `--labels` | шлях | GT-розмітка (`.json`, `.yaml` або папка з `.txt`) |
| `--task` | `det` / `seg` / `ignore` | Тип задачі |
| `--split` | `train` / `val` / `test` | Сплін датасету (тільки для `dataset`) |
| `--save-dir` | шлях | Куди зберігати результати |
| `--iou-thresh` | `0.0–1.0` | Поріг IoU для TP/FP (за замовч. `0.5`) |
| `--depth-threshold` | `0.0–1.0` | [depth] Поріг ближніх об'єктів (за замовч. `0.5`) |
| `--depth-invert` | flag | [depth] Вибирати ДАЛЕКІ об'єкти замість ближніх |
