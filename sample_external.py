"""
sample_external.py
==================
Демонстрація та базове тестування адаптерів для зовнішніх проектів:
  - IgnoreAdapter  (../ignore — FastSplitDetector)
  - InterfaceAdapter (../Video_Interface_detection_investigation)

Використання:
    python sample_external.py [--test-image PATH] [--test-video PATH]
                              [--train-videos V1 V2 ...]
                              [--mask-dir PATH]
                              [--detector geometric|yolo_interface|...]

Без аргументів запускає smoke-test (перевірку імпортів і конструкторів).
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


def _banner(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print('=' * 55)


# ─────────────────────────────────────────────────────────────────
# Тест 1: IgnoreAdapter smoke-test
# ─────────────────────────────────────────────────────────────────

def test_ignore_adapter_smoke(mask_dir=None):
    _banner("IgnoreAdapter — smoke test")
    from models.ignore_adapter import IgnoreAdapter

    adapter = IgnoreAdapter(mask_dir=mask_dir)
    print(f"  Тип:       {type(adapter).__name__}")
    print(f"  mask_dir:  {adapter.mask_dir}")
    print(f"  detector:  {adapter._detector}")

    # Запускаємо predict на порожньому (чорному) кадрі
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = adapter.predict(dummy)
    print(f"  predict на порожньому кадрі -> {result}")
    print("  ✅ IgnoreAdapter smoke-test пройдено")


# ─────────────────────────────────────────────────────────────────
# Тест 2: IgnoreAdapter.train() → predict() на реальному відео
# ─────────────────────────────────────────────────────────────────

def test_ignore_adapter_full(video_paths: list, mask_dir=None, test_image=None):
    _banner("IgnoreAdapter — повний тест (train + predict)")
    from models.ignore_adapter import IgnoreAdapter

    adapter = IgnoreAdapter(mask_dir=mask_dir)

    # Тренування якщо масок ще немає
    single_mask = adapter.mask_dir / "mask_single_cam.png"
    if not single_mask.exists():
        print("  Маски не знайдено — запускаємо тренування...")
        adapter.train(video_paths=video_paths, total_sample_count=400)
    else:
        print(f"  Маски вже є: {adapter.mask_dir} — пропускаємо train()")

    # Predict на тестовому зображенні або першому кадрі відео
    if test_image:
        result = adapter.predict(test_image)
        print(f"  predict на {test_image} -> masks={len(result['masks'])}, boxes={result['boxes']}")
        adapter.predict_and_save(test_image, save_dir="results/ignore")
    elif video_paths:
        cap = cv2.VideoCapture(video_paths[0])
        ret, frame = cap.read()
        cap.release()
        if ret:
            result = adapter.predict(frame)
            print(f"  predict на першому кадрі відео -> masks={len(result['masks'])}")
            if result['masks']:
                coverage = 100.0 * result['masks'][0].sum() / result['masks'][0].size
                print(f"  Покриття маскою: {coverage:.1f}%")
        else:
            print("  [WARN] Не вдалося прочитати перший кадр відео")

    print("  ✅ IgnoreAdapter повний тест завершено")


# ─────────────────────────────────────────────────────────────────
# Тест 3: InterfaceAdapter smoke-test
# ─────────────────────────────────────────────────────────────────

DETECTOR_CONFIGS = {
    "geometric": {
        "detector_module": "detectors.geometric_detector",
        "detector_class": "GeometricDetector",
    },
    "gmm": {
        "detector_module": "detectors.gmm_detector",
        "detector_class": "GMMDetector",
    },
    "mask": {
        "detector_module": "detectors.mask_detector",
        "detector_class": "MaskDetector",
    },
    "yolo_interface": {
        "detector_module": "detectors.yolo_interface_detector",
        "detector_class": "YOLOInterfaceDetector",
        "model_path": "../Video_Interface_detection_investigation/models/best.pt",
        "confidence": 0.25,
        "iou_threshold": 0.45,
        "device": "cpu",
    },
}


def test_interface_adapter_smoke(detector_name: str = "geometric", test_image=None):
    _banner(f"InterfaceAdapter — smoke test [{detector_name}]")
    from models.interface_adapter import InterfaceAdapter

    cfg = DETECTOR_CONFIGS.get(detector_name)
    if cfg is None:
        print(f"  [WARN] Детектор '{detector_name}' не знайдено. Доступні: {list(DETECTOR_CONFIGS)}")
        return

    try:
        adapter = InterfaceAdapter(config=cfg)
    except Exception as e:
        print(f"  [ERROR] Не вдалося ініціалізувати {detector_name}: {e}")
        return

    # Predict на dummy-кадрі
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = adapter.predict(dummy)
    print(f"  predict на порожньому кадрі -> boxes={result['boxes']}, classes={result['classes']}")

    # Predict на реальному зображенні (якщо передано)
    if test_image:
        result = adapter.predict(test_image)
        print(f"  predict на {test_image} -> boxes={len(result['boxes'])}")
        adapter.predict_and_save(test_image, save_dir=f"results/interface_{detector_name}")

    print(f"  ✅ InterfaceAdapter [{detector_name}] smoke-test пройдено")


# ─────────────────────────────────────────────────────────────────
# Точка входу
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Тестування зовнішніх адаптерів yolo_evaluator"
    )
    parser.add_argument("--test-image", type=str, default=None,
                        help="Шлях до тестового зображення")
    parser.add_argument("--test-video", type=str, default=None,
                        help="Шлях до тестового відео (для evaluate_video)")
    parser.add_argument("--train-videos", nargs="+", default=[],
                        help="Відео для тренування IgnoreAdapter")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Папка для збереження масок IgnoreAdapter")
    parser.add_argument("--detector", type=str, default="geometric",
                        choices=list(DETECTOR_CONFIGS.keys()),
                        help="Детектор для InterfaceAdapter")
    args = parser.parse_args()

    # ── IgnoreAdapter ──────────────────────────────────────────────
    test_ignore_adapter_smoke(mask_dir=args.mask_dir)

    if args.train_videos:
        test_ignore_adapter_full(
            video_paths=args.train_videos,
            mask_dir=args.mask_dir,
            test_image=args.test_image,
        )

    # ── InterfaceAdapter ───────────────────────────────────────────
    test_interface_adapter_smoke(
        detector_name=args.detector,
        test_image=args.test_image,
    )

    print("\n🎉 Усі тести завершено.")


if __name__ == "__main__":
    main()
