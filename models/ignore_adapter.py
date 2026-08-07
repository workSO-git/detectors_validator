"""
IgnoreAdapter
=============
Адаптер для інтеграції FastSplitDetector та AutoMixedLayoutTrainer
(папка ../ignore) у фреймворк yolo_evaluator.

Двофазний підхід:
  1. ОФЛАЙН: train(video_paths) — запускає AutoMixedLayoutTrainer,
     генерує маски single_cam/dual_cam і зберігає їх у mask_dir.
     Треба виконати один раз перед оцінкою.
  2. ОНЛАЙН:  predict(image)  — FastSplitDetector обирає відповідну
     маску і повертає її у стандартному форматі BaseModel.

Шлях до проекту зчитується з config_paths.yaml (ключ external_paths.ignore).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

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


class IgnoreAdapter(BaseModel):
    """
    Адаптер для FastSplitDetector з проекту ignore.

    Parameters
    ----------
    mask_dir : str | Path
        Папка, де зберігаються/будуть збережені маски
        (mask_single_cam.png, mask_dual_cam.png).
        Якщо не вказано — використовується папка 'drone_masks'
        всередині проекту ignore.
    split_threshold : float
        Поріг для розпізнавання dual-cam режиму (за замовчуванням 35.0).
    """

    def __init__(
        self,
        mask_dir: Optional[str | Path] = None,
        split_threshold: float = 35.0,
    ):
        # Додаємо шлях до зовнішнього проекту у sys.path
        ignore_root = _resolve_external_path("ignore")
        if str(ignore_root) not in sys.path:
            sys.path.insert(0, str(ignore_root))

        # Імпорт після додавання до sys.path
        try:
            from detector import FastSplitDetector  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                f"Не вдалося імпортувати FastSplitDetector з {ignore_root}: {e}"
            ) from e

        self._FastSplitDetector = FastSplitDetector
        self._ignore_root = ignore_root
        self._split_threshold = split_threshold

        # Визначаємо папку масок
        if mask_dir is None:
            mask_dir = ignore_root / "drone_masks"
        self.mask_dir = Path(mask_dir)

        self._detector: Optional[FastSplitDetector] = None
        self._init_detector()

    def _init_detector(self):
        """Ініціалізує FastSplitDetector якщо маски вже існують."""
        single = self.mask_dir / "mask_single_cam.png"
        dual = self.mask_dir / "mask_dual_cam.png"
        if single.exists() or dual.exists():
            self._detector = self._FastSplitDetector(mask_dir=str(self.mask_dir))
            print(f"[IgnoreAdapter] FastSplitDetector ініціалізовано з масками: {self.mask_dir}")
        else:
            print(
                f"[IgnoreAdapter] Маски не знайдено в '{self.mask_dir}'. "
                "Викличте .train(video_paths) для їх створення."
            )

    # ── Офлайн: генерація масок ───────────────────────────────────────────────

    def train(
        self,
        video_paths: list[str],
        total_sample_count: int = 800,
        split_threshold: float = 35.0,
    ) -> None:
        """
        Офлайн фаза: генерує маски single_cam/dual_cam з відеофайлів.
        Зберігає результат у self.mask_dir.

        Parameters
        ----------
        video_paths : list[str]
            Список шляхів до відеофайлів для навчання.
        total_sample_count : int
            Загальна кількість зразків (рівномірно розподіляється між відео).
        split_threshold : float
            Поріг розпізнавання dual-cam режиму.
        """
        try:
            from detector import AutoMixedLayoutTrainer  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                f"Не вдалося імпортувати AutoMixedLayoutTrainer з {self._ignore_root}: {e}"
            ) from e

        print(f"[IgnoreAdapter] Починаємо тренування масок...")
        print(f"  Відео:    {len(video_paths)} файлів")
        print(f"  Зразків:  {total_sample_count}")
        print(f"  Mask dir: {self.mask_dir}")

        trainer = AutoMixedLayoutTrainer(output_dir=str(self.mask_dir))
        trainer.process_multiple_videos(
            video_paths=video_paths,
            total_sample_count=total_sample_count,
            split_threshold=split_threshold,
        )
        print("[IgnoreAdapter] Тренування завершено. Ініціалізуємо детектор...")
        self._init_detector()

    # ── Онлайн: predict ───────────────────────────────────────────────────────

    def predict(self, image_path_or_array, conf_threshold: float = 0.25) -> dict:
        """
        Визначає OSD-маску для зображення/кадру.

        Returns
        -------
        dict
            Стандартний формат BaseModel:
            {
                'masks':   [binary_mask] або [],  # uint8, shape (H, W)
                'boxes':   [],
                'classes': [0] або [],
            }
        """
        if self._detector is None:
            return {"masks": [], "boxes": [], "classes": []}

        # Завантажуємо зображення якщо передано шлях
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        else:
            img = image_path_or_array

        if img is None:
            return {"masks": [], "boxes": [], "classes": []}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = self._detector.get_ignore_mask(gray, split_threshold=self._split_threshold)

        if mask is None or np.count_nonzero(mask) == 0:
            return {"masks": [], "boxes": [], "classes": []}

        # Нормалізуємо маску до бінарного uint8 (0 або 1)
        h, w = img.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask > 0).astype(np.uint8)

        return {"masks": [binary_mask], "boxes": [], "classes": [0]}

    def predict_and_save(self, image_path, save_dir=None) -> None:
        """Запускає інференс та зберігає візуалізацію."""
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[IgnoreAdapter] Файл не знайдено: {image_path}")
            return

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[IgnoreAdapter] Не вдалося відкрити зображення: {image_path}")
            return

        result = self.predict(img)
        masks = result.get("masks", [])

        display = img.copy()
        if masks:
            mask = masks[0]
            red_overlay = np.zeros_like(display)
            red_overlay[:] = (30, 30, 220)
            idx = mask == 1
            display[idx] = cv2.addWeighted(display, 0.3, red_overlay, 0.7, 0)[idx]

        mode = getattr(self._detector, "current_mode", "unknown")
        cv2.putText(
            display,
            f"Mode: {mode} | Mask px: {int(np.sum(masks[0])) if masks else 0}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = Path(save_dir) / image_path.name
            cv2.imwrite(str(out_path), display)
            print(f"[IgnoreAdapter] ✅ Збережено: {out_path}")
        else:
            print("[IgnoreAdapter] ✅ Інференс завершено.")
