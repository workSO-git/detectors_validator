"""
DINOv2MlpModel
==============
Адаптер для моделі DINOv2-Small + MLP Head (best_dinov2_mlp_model.pth).

Архітектура (з dinov2_architecture_comparison.md):
  - Backbone: dinov2_vits14 (ViT-Small, patch_size=14, embed_dim=384)
  - MLP Head: Conv2d(384→256) → BN → ReLU → Conv2d(256→64) → BN → ReLU → Conv2d(64→1)
  - Вхід: 518×518 (ділиться на 14), ImageNet-нормалізація
  - Вихід: бінарна маска сегментації неба

Використання в main.py:
  python main.py --mode single --model-type dinov2 --model ../best_dinov2_mlp_model.pth ...
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .base_model import BaseModel


# ──────────────────────────────────────────────────────────────────────────────
# Архітектура моделі — повинна збігатися з тим, як вона тренувалась
# ──────────────────────────────────────────────────────────────────────────────

class _MlpHead(nn.Sequential):
    """
    MLP Head поверх патч-токенів DINOv2.
    384 ознаки → Conv2d(256) → BN → ReLU → Conv2d(64) → BN → ReLU → Conv2d(1)

    Наслідує nn.Sequential напряму, щоб ключі state_dict мали вигляд
    'head.0.weight', 'head.1.weight' … (а не 'head.head.0.weight').
    """

    def __init__(self, in_channels: int = 384, num_classes: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )


class _DinoV2SegModel(nn.Module):
    """
    Повна модель: DINOv2-Small backbone (dinov2_vits14) + MLP Head.
    Backbone завантажується через torch.hub з Meta AI.
    """

    PATCH_SIZE = 14
    EMBED_DIM = 384  # dinov2_vits14

    def __init__(self):
        super().__init__()
        # Завантажуємо backbone через torch.hub (може закешуватись локально)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=False,  # ваги завантажуємо зі свого .pth файлу
        )
        self.head = _MlpHead(in_channels=self.EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W), де H і W кратні PATCH_SIZE=14
        Повертає: (B, 1, H, W) — логіти для сегментації
        """
        B, C, H, W = x.shape
        h_patches = H // self.PATCH_SIZE
        w_patches = W // self.PATCH_SIZE

        # Отримуємо патч-токени: (B, h_patches*w_patches, embed_dim)
        features = self.backbone.get_intermediate_layers(x, n=1)[0]  # (B, N, D)

        # Reshape до (B, D, h_patches, w_patches)
        features = features.permute(0, 2, 1).reshape(B, self.EMBED_DIM, h_patches, w_patches)

        # MLP Head → логіти (B, 1, h_patches, w_patches)
        logits = self.head(features)

        # Bilinear upsample до оригінального розміру
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Адаптер для yolo_evaluator
# ──────────────────────────────────────────────────────────────────────────────

class DINOv2MlpModel(BaseModel):
    """
    Адаптер для best_dinov2_mlp_model.pth у фреймворку yolo_evaluator.

    Parameters
    ----------
    model_path : str | Path
        Шлях до файлу .pth з вагами (state_dict або повна модель).
    task : str
        Тип задачі (завжди 'seg').
    input_size : tuple[int, int]
        (H, W) — розмір входу для мережі. Обидва числа мають ділитися на 14.
        За замовчуванням 518×518, як при навчанні.
    conf_threshold : float
        Поріг Sigmoid для бінаризації маски (за замовчуванням 0.5).
    """

    # ImageNet mean/std — обов'язково, оскільки DINOv2 тренувався на ImageNet
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str | Path,
        task: str = "seg",
        input_size: tuple[int, int] = (518, 518),
        conf_threshold: float = 0.5,
    ):
        self.task = task
        self.input_size = input_size      # (H, W)
        self.conf_threshold = conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DINOv2MlpModel] Завантаження моделі з {model_path} на {self.device}...")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"[DINOv2MlpModel] Файл не знайдено: {model_path}")

        # Ініціалізуємо архітектуру
        self.model = _DinoV2SegModel()

        # Завантажуємо ваги
        state = torch.load(str(model_path), map_location=self.device)

        # Підтримуємо як state_dict, так і повну модель
        if isinstance(state, dict):
            # Якщо збережено у checkpoint-форматі {'model_state_dict': ...}
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state, strict=True)
        else:
            # Збережено як повний об'єкт моделі — малоймовірно, але на всяк випадок
            self.model = state

        self.model.to(self.device)
        self.model.eval()
        print("[DINOv2MlpModel] Модель готова до інференсу ✅")

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """BGR ndarray → нормалізований тензор (1, 3, H, W)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size[1], self.input_size[0]))  # (W, H)
        img_f32 = img_resized.astype(np.float32) / 255.0
        # ImageNet нормалізація
        img_norm = (img_f32 - self._IMAGENET_MEAN) / self._IMAGENET_STD
        # HWC → CHW → BCHW
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, image_path_or_array, conf_threshold: float | None = None) -> dict:
        """
        Сегментує небо на зображенні.

        Parameters
        ----------
        image_path_or_array : str | Path | np.ndarray
            Шлях до зображення або BGR ndarray.
        conf_threshold : float | None
            Якщо передано — перекриває self.conf_threshold для цього виклику.

        Returns
        -------
        dict з ключами 'masks', 'boxes', 'classes' (стандартний формат BaseModel).
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        else:
            img = image_path_or_array

        if img is None:
            return {"masks": [], "boxes": [], "classes": []}

        h, w = img.shape[:2]
        tensor = self._preprocess(img)

        with torch.no_grad():
            logits = self.model(tensor)               # (1, 1, H_in, W_in)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        # Бінарна маска у просторі входу мережі
        mask_net = (probs >= conf_threshold).astype(np.uint8)

        # Повертаємо до оригінального розміру
        mask = cv2.resize(mask_net, (w, h), interpolation=cv2.INTER_NEAREST)

        return {
            "masks":   [mask],
            "boxes":   [],
            "classes": [0],
        }

    # ── Predict & Save ────────────────────────────────────────────────────────

    def predict_and_save(self, image_path, save_dir=None) -> None:
        """Запускає інференс та зберігає порівняльну візуалізацію."""
        image_path = Path(image_path)
        orig_bgr = cv2.imread(str(image_path))
        if orig_bgr is None:
            print(f"[DINOv2MlpModel] Не вдалося відкрити: {image_path}")
            return

        result = self.predict(orig_bgr)
        masks = result.get("masks", [])
        mask = masks[0] if masks else np.zeros(orig_bgr.shape[:2], dtype=np.uint8)

        overlay = orig_bgr.copy()
        # Напівпрозорий синій оверлей для неба
        blue = np.zeros_like(overlay)
        blue[:] = (255, 100, 0)  # BGR: синій
        idx = mask == 1
        overlay[idx] = (overlay[idx] * 0.45 + blue[idx] * 0.55).astype(np.uint8)

        vis = np.concatenate([orig_bgr, overlay], axis=1)

        if save_dir:
            save_path = Path(save_dir) / image_path.name
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
            print(f"[DINOv2MlpModel] ✅ Збережено: {save_path}")
        else:
            print("[DINOv2MlpModel] ✅ Інференс завершено.")
