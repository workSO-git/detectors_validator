# COMMANDS — Команди для запуску оцінки моделей

Всі команди запускаються з кореневої папки проекту:
```
cd C:\Users\Sasha\projects\CV\yolo_evaluator
```

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
  --model "C:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt" `
  --source "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4" `
  --labels "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json" `
  --task det
```

### YOLO Segmentation (seg)
```powershell
python main.py --mode video_eval `
  --model-type yolo `
  --model "C:\Users\Sasha\projects\CV\models_extracted\models\best.pt" `
  --source "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4" `
  --labels "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json" `
  --task seg
```

### EMA Detector (Interface, детектор зон ігнорування)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.ema_detector', 'detector_class': 'EMADetector'}" `
  --source "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4" `
  --labels "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json" `
  --task det
```

### GMM Detector (Interface)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.gmm_detector', 'detector_class': 'GMMDetector'}" `
  --source "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4" `
  --labels "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json" `
  --task det
```

### Geometric Detector (Interface)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.geometric_detector', 'detector_class': 'GeometricDetector'}" `
  --source "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4" `
  --labels "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json" `
  --task det
```

### YOLO Interface Detector (через VIDI-обгортку)
```powershell
python main.py --mode video_eval `
  --model-type interface `
  --model "{'detector_module': 'detectors.yolo_interface_detector', 'detector_class': 'YOLOInterfaceDetector', 'model_path': 'C:\\Users\\Sasha\\projects\\CV\\det_pipeline\\runs\\drone_det_n\\weights\\best.pt'}" `
  --source "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4" `
  --labels "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json" `
  --task det
```

### IgnoreAdapter (навчається прямо на відео перед оцінкою)
```powershell
python main.py --mode video_eval `
  --model-type ignore `
  --model "{'detector_module': 'detectors.ema_detector', 'detector_class': 'EMADetector'}" `
  --source "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4" `
  --labels "C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json" `
  --task ignore
```

---

## 📂 Тестування на ДАТАСЕТАХ (`--mode dataset`)

### YOLO Object Detection — датасет детекції
```powershell
python main.py --mode dataset `
  --model-type yolo `
  --model "C:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt" `
  --source "C:\Users\Sasha\projects\CV\det_dataset" `
  --task det `
  --split val
```

### YOLO Segmentation — датасет сегментації
```powershell
python main.py --mode dataset `
  --model-type yolo `
  --model "C:\Users\Sasha\projects\CV\models_extracted\models\best.pt" `
  --source "C:\Users\Sasha\projects\CV\yolo_seg_dataset\yolo_seg_dataset" `
  --task seg `
  --split val
```

### YOLO Detection — датасет пайплайну (train split)
```powershell
python main.py --mode dataset `
  --model-type yolo `
  --model "C:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt" `
  --source "C:\Users\Sasha\projects\CV\det_dataset" `
  --task det `
  --split train
```

### EMA Detector — на датасеті пайплайну (через JSON розмітку)
```powershell
python main.py --mode dataset `
  --model-type interface `
  --model "{'detector_module': 'detectors.ema_detector', 'detector_class': 'EMADetector'}" `
  --source "C:\Users\Sasha\projects\CV\det_dataset" `
  --labels "C:\Users\Sasha\projects\CV\db_task\merged.json" `
  --task det `
  --split val
```

---

## 🖼️ Тестування одного ЗОБРАЖЕННЯ (`--mode single`)

### YOLO (одне фото)
```powershell
python main.py --mode single `
  --model-type yolo `
  --model "C:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt" `
  --source "C:\шлях\до\фото.jpg" `
  --task det
```

### EMA Detector (одне фото)
```powershell
python main.py --mode single `
  --model-type interface `
  --model "{'detector_module': 'detectors.ema_detector', 'detector_class': 'EMADetector'}" `
  --source "C:\шлях\до\фото.jpg" `
  --task det
```

---

## 📌 Шляхи до ресурсів

| Ресурс | Шлях |
|--------|------|
| YOLO nano det модель | `C:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt` |
| YOLO seg модель | `C:\Users\Sasha\projects\CV\models_extracted\models\best.pt` |
| Відео для тесту | `C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043.mp4` |
| GT для відео (JSON) | `C:\Users\Sasha\projects\CV\drone-videos\drone-videos\signal-2026-07-02-15-19-19-043_gt.json` |
| GT загальний (JSON) | `C:\Users\Sasha\projects\CV\db_task\merged.json` |
| Датасет детекції | `C:\Users\Sasha\projects\CV\det_dataset` |
| Датасет сегментації | `C:\Users\Sasha\projects\CV\yolo_seg_dataset\yolo_seg_dataset` |
| VIDI проект (інтерфейс) | `C:\Users\Sasha\projects\CV\Video_Interface_detection_investigation` |

---

## 🔧 Параметри `main.py`

| Параметр | Значення | Опис |
|----------|----------|------|
| `--mode` | `dataset` / `single` / `video_eval` | Режим оцінки |
| `--model-type` | `yolo` / `interface` / `ignore` | Тип адаптера |
| `--model` | шлях або dict | Шлях до `.pt` або конфіг детектора |
| `--source` | шлях | Папка датасету, файл відео або фото |
| `--labels` | шлях | GT розмітка (.json, .yaml, або папка) |
| `--task` | `det` / `seg` / `ignore` | Тип задачі |
| `--split` | `train` / `val` / `test` | Сплітдатасету (тільки для `dataset`) |
| `--save-dir` | шлях | Куди зберігати результати |
| `--iou-thresh` | 0.0–1.0 | Поріг IoU для TP/FP (за замовч. 0.5) |
