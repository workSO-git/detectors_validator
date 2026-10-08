# Algorithm Evaluator (Detectors Validator)

Веб-застосунок та CLI для оцінки алгоритмів комп'ютерного зору (сегментація, детекція). Підтримує YOLO, SMP/ResNet, DINOv2, Depth-Anything V2, а також детектори зовнішніх проектів через адаптери.

---

## Основний функціонал

*   **Підтримка фото та відео:** аналіз окремих зображень і відеопотоків.
*   **Інтерактивний веб-UI:** завантаження файлів drag-and-drop, перегляд результатів у реальному часі.
*   **Різні режими перегляду:** Оригінал / Маски / Бокси / Side-by-side.
*   **Автоматичний пошук розмітки:** вкажіть `dataset.yaml` або `.json` — система знайде відповідні GT-файли.
*   **CLI-режим:** запуск оцінки без UI через `main.py` (dataset, single, video_eval).

---

## Встановлення

### 1. Клонуйте репозиторій

```bash
git clone <repo-url>
cd yolo_evaluator
```

### 2. Встановіть Python-залежності

```bash
pip install -r requirements.txt
```

> **PyTorch потрібен окремо** — встановіть відповідну версію з CUDA або CPU:
> ```bash
> # Приклад для CUDA 12.1:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> # Або CPU-варіант:
> pip install torch torchvision
> ```

### 3. Налаштуйте середовище (.env)

```bash
copy .env.example .env    # Windows
# або
cp .env.example .env      # Linux/macOS
```

Відкрийте `.env` і вкажіть шляхи до ваших файлів ваг:

```dotenv
YOLO_DET_MODEL=C:/path/to/yolo_det/weights/best.pt
YOLO_SEG_MODEL=C:/path/to/yolo_seg/weights/best.pt
SMP_SKY_MODEL=C:/path/to/best_sky_model.pth
# ... (решта дивіться у .env.example)
```

### 4. Налаштуйте зовнішні проекти (config_paths.yaml)

Файл `config_paths.yaml` містить відносні шляхи до зовнішніх проектів (відносно кореня `yolo_evaluator/`):

```yaml
external_paths:
  ignore: "../ignore"                                      # FastSplitDetector
  interface_investigation: "../Video_Interface_detection_investigation"  # VIDI
  dinov2_mlp_model: "../best_dinov2_mlp_model.pth"
```

Якщо ці проекти розташовані у нестандартних місцях — відкоригуйте шляхи у `config_paths.yaml`.

### 5. Запустіть веб-сервер

```bash
python -m uvicorn web_app.server:app --host 127.0.0.1 --port 8000
```

Відкрийте [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Структура проекту

```
yolo_evaluator/
├── main.py                  — CLI-точка входу (dataset / single / video_eval)
├── evaluator.py             — ядро оцінки: GT-завантаження, порівняння, метрики
├── metrics.py               — IoU, Precision, Recall, F1, Jitter, Flicker
├── config_paths.yaml        — шляхи до зовнішніх проектів (відносні)
├── .env                     — локальні шляхи до ваг моделей (НЕ в git)
├── .env.example             — шаблон .env для нових розробників
├── requirements.txt         — Python-залежності
├── models/
│   ├── base_model.py        — базовий клас BaseModel
│   ├── base_video_model.py  — базовий клас для відеомоделей
│   ├── yolo_model.py        — адаптер YOLO (ultralytics)
│   ├── smp_model.py         — адаптер SMP/ResNet (.pth / .pt / .onnx)
│   ├── dinov2_model.py      — адаптер DINOv2 + MLP Head
│   ├── depth_anything_model.py — адаптер Depth-Anything V2 (HuggingFace)
│   ├── ignore_adapter.py    — адаптер FastSplitDetector (зовнішній проект)
│   ├── interface_adapter.py — адаптер VIDI-детекторів (зовнішній проект)
│   ├── horizon_adapter.py   — алгоритмічна детекція горизонту (OpenCV)
│   └── tracker_adapter.py   — трекінг горизонту (Optical Flow)
└── web_app/
    ├── server.py            — FastAPI сервер, REST API, WebSocket
    └── static/              — веб-інтерфейс (HTML, JS, CSS)
```

---

## Підтримувані моделі (`--model-type`)

| `--model-type` | Опис | `--model` |
|---|---|---|
| `yolo` | YOLO Detection/Segmentation (ultralytics) | шлях до `.pt` |
| `reznet` | SMP UNet/ResNet (`.pth` / `.pt` TorchScript / `.onnx`) | шлях до файлу |
| `dinov2` | DINOv2-Small + MLP Head | шлях до `.pth` |
| `depth` | Depth-Anything V2 (HuggingFace, auto-download) | `none` або шлях до локального кешу |
| `interface` | Довільний детектор з VIDI-проекту | `module:Class` або dict-рядок |
| `ignore` | FastSplitDetector (маска ігнорування) | шлях до папки масок або `none` |
| `horizon` | Алгоритмічна детекція горизонту (OpenCV) | `algorithm` |
| `tracker` | Трекінг горизонту (Optical Flow) | `algorithm` |

---

## Як рахуються метрики

### Метрики детекції / сегментації

Використовується **жадібне співставлення (Greedy Matching)**:

1.  IoU (маски або бокси) між кожною передбаченою та GT-областю.
2.  **TP** — пара з IoU ≥ порогу (за замовч. `0.5`).
3.  **FP** — передбачення без відповідного GT.
4.  **FN** — GT-об'єкт, не знайдений моделлю.

На їх основі: **Precision** = `TP / (TP+FP)`, **Recall** = `TP / (TP+FN)`, **F1** = гармонійне середнє.

### Темпоральні метрики (відео)

Ковзне вікно на останні **50 кадрів**:

*   **Temporal Jitter** — евклідова відстань між центроїдами суміжних кадрів (px). Ідеально < 5px.
*   **Flicker Rate** — частка кадрів зі зміною статусу детекції (є/немає). Ідеально < 5%.

---

## Технологічний стек

*   **Backend:** Python 3.10+, FastAPI, Uvicorn, WebSockets
*   **CV / DL:** OpenCV, PyTorch, Ultralytics (YOLO), HuggingFace Transformers, SMP, DINOv2
*   **Frontend:** HTML5, Vanilla JavaScript, CSS
