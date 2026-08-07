# Init for models package
from .base_model import BaseModel
from .base_video_model import BaseVideoModel
from .yolo_model import YoloModel
from .ignore_adapter import IgnoreAdapter
from .interface_adapter import InterfaceAdapter

__all__ = [
    "BaseModel",
    "BaseVideoModel",
    "YoloModel",
    "IgnoreAdapter",
    "InterfaceAdapter",
]
