# Init for models package
from .base_model import BaseModel
from .base_video_model import BaseVideoModel

try:
    from .yolo_model import YoloModel
except Exception:
    YoloModel = None

try:
    from .ignore_adapter import IgnoreAdapter
except Exception:
    IgnoreAdapter = None

try:
    from .interface_adapter import InterfaceAdapter
except Exception:
    InterfaceAdapter = None

try:
    from .depth_anything_model import DepthAnythingModel
except Exception:
    DepthAnythingModel = None

try:
    from .smp_model import SmpModel
except Exception:
    SmpModel = None

try:
    from .dinov2_model import DINOv2MlpModel
except Exception:
    DINOv2MlpModel = None

try:
    from .camera_adapter import CameraAdapter
except Exception:
    CameraAdapter = None

