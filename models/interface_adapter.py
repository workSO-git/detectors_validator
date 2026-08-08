"""
InterfaceAdapter
================
Адаптер для детекторів з проекту Video_Interface_detection_investigation
(VIDI) у фреймворк yolo_evaluator.

Підтримує динамічне завантаження будь-якого класу детектора через
importlib — детектор вказується у config-словнику.

Усі детектори VIDI наслідують BaseDetector і мають метод detect(image),
що повертає список bounding boxes у форматі xyxy (абсолютні пікселі).

Шлях до проекту зчитується з config_paths.yaml (ключ
external_paths.interface_investigation).

Приклад використання:
---------------------
    cfg = {
        "detector_module": "detectors.geometric_detector",
        "detector_class":  "GeometricDetector",
        # ...параметри конкретного детектора...
    }
    adapter = InterfaceAdapter(cfg)
    result = adapter.predict(frame)
    # result = {'masks': [], 'boxes': [[x1,y1,x2,y2], ...], 'classes': [0, ...]}

Доступні детектори (з detectors/__init__.py):
    - YOLOWorldDetector   (detectors.yolo_world_detector)
    - YOLOCustomDetector  (detectors.yolo_custom_detector)
    - YOLOInterfaceDetector (detectors.yolo_interface_detector) [не в __init__]
    - EASTDetector        (detectors.east_detector)
    - GMMDetector         (detectors.gmm_detector)
    - MaskDetector        (detectors.mask_detector)
    - GeometricDetector   (detectors.geometric_detector)
"""

from __future__ import annotations

import sys
import os
import importlib
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
import yaml

from .base_model import BaseModel


def _load_paths_config() -> dict:
    """Зчитує config_paths.yaml відносно папки yolo_evaluator."""
    cfg_path = Path(__file__).parent.parent / "config_paths.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_paths.yaml не знайдено: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_external_path(key: str) -> Path:
    """Повертає абсолютний шлях до зовнішнього проекту за ключем."""
    cfg = _load_paths_config()
    rel = cfg.get("external_paths", {}).get(key)
    if rel is None:
        raise KeyError(f"Ключ '{key}' відсутній у config_paths.yaml -> external_paths")
    base = Path(__file__).parent.parent  # yolo_evaluator/
    return (base / rel).resolve()


class InterfaceAdapter(BaseModel):
    """
    Адаптер для будь-якого детектора з VIDI-проекту.

    Parameters
    ----------
    config : dict
        Словник конфігурації з обов'язковими ключами:
          - detector_module (str): модуль відносно кореня VIDI-проекту,
            наприклад "detectors.geometric_detector"
          - detector_class (str): назва класу, наприклад "GeometricDetector"
        Усі інші ключі передаються безпосередньо в конструктор детектора.
        Якщо детектор приймає dict (старий стиль), передається весь config.
        Якщо приймає keyword arguments — передається **config (мінус
        detector_module та detector_class).

    Raises
    ------
    KeyError
        Якщо у config відсутні detector_module або detector_class.
    ImportError
        Якщо модуль/клас не вдається знайти.
    """

    def __init__(self, config: dict):
        vidi_root = _resolve_external_path("interface_investigation")
        if str(vidi_root) not in sys.path:
            sys.path.insert(0, str(vidi_root))

        self._vidi_root = vidi_root

        module_name = config.get("detector_module")
        class_name = config.get("detector_class")

        if not module_name:
            raise KeyError("config повинен містити 'detector_module'")
        if not class_name:
            raise KeyError("config повинен містити 'detector_class'")

        # Динамічне завантаження класу
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Не вдалося імпортувати модуль '{module_name}' "
                f"з {vidi_root}: {e}"
            ) from e

        cls = getattr(mod, class_name, None)
        if cls is None:
            available = [n for n in dir(mod) if not n.startswith("_")]
            raise ImportError(
                f"Клас '{class_name}' не знайдено у '{module_name}'. "
                f"Доступні: {available}"
            )

        # Ініціалізуємо детектор
        # Старий стиль приймає dict, BaseDetector — може приймати name + device
        detector_kwargs = {k: v for k, v in config.items()
                          if k not in ("detector_module", "detector_class")}
        try:
            # Спробуємо передати config як dict (стиль YOLOInterfaceDetector)
            self._detector = cls(config=config)
        except TypeError:
            try:
                # Спробуємо kwargs (сучасний стиль)
                self._detector = cls(**detector_kwargs)
            except TypeError:
                # Без аргументів
                self._detector = cls()

        # Викликаємо load() якщо детектор ще не готовий
        if hasattr(self._detector, "is_ready") and not self._detector.is_ready:
            if hasattr(self._detector, "load"):
                self._detector.load()

        print(
            f"[InterfaceAdapter] Завантажено: {class_name} з {module_name}\n"
            f"  Детектор: {self._detector}"
        )
        
    @property
    def warming_up(self):
        """Returns True if the underlying detector is still learning/warming up (e.g., GMM)."""
        if hasattr(self._detector, 'frames_processed') and hasattr(self._detector, 'learning_frames'):
            return self._detector.frames_processed < self._detector.learning_frames
        return False

    def set_video_mode(self, is_video: bool):
        if hasattr(self._detector, 'set_video_mode'):
            self._detector.set_video_mode(is_video)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, image_path_or_array, conf_threshold: float = 0.25) -> dict:
        """
        Запускає детектор і повертає результат у стандартному форматі.

        Returns
        -------
        dict
            {
                'masks':   [],
                'boxes':   [[x1, y1, x2, y2], ...],  # абсолютні пікселі
                'classes': [0, ...],
            }
        """
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        else:
            img = image_path_or_array

        if img is None:
            return {"masks": [], "boxes": [], "classes": []}

        # Allow detectors to bypass the standard detect() if they provide their own predict()
        # This is useful for detectors that want to return masks or custom dicts (e.g. EMADetector)
        if hasattr(self._detector, "predict"):
            try:
                return self._detector.predict(img, conf_threshold=conf_threshold)
            except Exception as e:
                print(f"[InterfaceAdapter] Помилка predict(): {e}")
                return {"masks": [], "boxes": [], "classes": []}

        try:
            raw = self._detector.detect(img)
        except Exception as e:
            print(f"[InterfaceAdapter] Помилка detect(): {e}")
            return {"masks": [], "boxes": [], "classes": []}

        # Нормалізуємо вихід у список [[x1,y1,x2,y2], ...]
        boxes: list[list[int]] = []
        if raw:
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    boxes.append([int(v) for v in item])

        return {
            "masks": [],
            "boxes": boxes,
            "classes": [0] * len(boxes),
        }

    # ── predict_and_save ──────────────────────────────────────────────────────

    def predict_and_save(self, image_path, save_dir=None) -> None:
        """Запускає інференс та зберігає візуалізацію bounding boxes."""
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[InterfaceAdapter] Файл не знайдено: {image_path}")
            return

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[InterfaceAdapter] Не вдалося відкрити зображення: {image_path}")
            return

        result = self.predict(img)
        boxes = result.get("boxes", [])
        display = img.copy()

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 80), 2)
            cv2.putText(
                display, "interface",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1,
            )

        cv2.putText(
            display,
            f"Detections: {len(boxes)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2,
        )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = Path(save_dir) / image_path.name
            cv2.imwrite(str(out_path), display)
            print(f"[InterfaceAdapter] ✅ Збережено: {out_path}")
        else:
            print("[InterfaceAdapter] ✅ Інференс завершено.")
