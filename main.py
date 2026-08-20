import argparse
from pathlib import Path
from evaluator import GenericEvaluator

def get_model_adapter(model_type, model_path, task, _depth_threshold=0.5, _depth_invert=False):
    """Factory function to instantiate the correct model adapter."""
    adapter = None
    if model_type == 'yolo':
        from models.yolo_model import YoloModel
        adapter = YoloModel(model_path, task=task)
    elif model_type in ['reznet', 'resnet']:
        from models.smp_model import SmpModel
        adapter = SmpModel(model_path, task=task)
    elif model_type == 'ignore':
        from models.ignore_adapter import IgnoreAdapter
        # For ignore, model_path can be used as mask_dir or omitted (use default)
        mask_dir = model_path if model_path and model_path.lower() != 'none' else None
        adapter = IgnoreAdapter(mask_dir=mask_dir)
    elif model_type == 'interface':
        from models.interface_adapter import InterfaceAdapter
        # If the string looks like a dictionary, parse it safely
        if isinstance(model_path, str) and model_path.strip().startswith('{') and model_path.strip().endswith('}'):
            import ast
            try:
                model_path = ast.literal_eval(model_path)
            except (ValueError, SyntaxError):
                pass
                
        if isinstance(model_path, dict):
            adapter = InterfaceAdapter(model_path)
        else:
            if not model_path or ':' not in model_path:
                raise ValueError("For interface models, provide --model as 'module:class' or a valid dict string")
            mod, cls = model_path.split(':', 1)
            adapter = InterfaceAdapter({'detector_module': mod, 'detector_class': cls})
    elif model_type == 'depth':
        from models.depth_anything_model import DepthAnythingModel
        import argparse
        # model_path can be a local cache dir or None/'none' to use HuggingFace auto-download
        local_path = model_path if (model_path and model_path.lower() != 'none') else None
        adapter = DepthAnythingModel(
            model_path=local_path,
            task='seg',
            depth_threshold=_depth_threshold,
            invert=_depth_invert,
        )
    elif model_type == 'horizon':
        from models.horizon_adapter import HorizonAdapter
        adapter = HorizonAdapter(task=task)
    elif model_type == 'tracker':
        from models.tracker_adapter import TrackerAdapter
        adapter = TrackerAdapter(task=task)
    elif model_type == 'dinov2':
        from models.dinov2_model import DINOv2MlpModel
        adapter = DINOv2MlpModel(model_path=model_path, task=task)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    if adapter is not None:
        adapter.task = task
    return adapter

def main():
    parser = argparse.ArgumentParser(description="Generic Model Evaluator (mIoU, Precision, Recall, F1, Jitter, Flicker)")
    parser.add_argument('--mode', type=str, required=True, choices=['dataset', 'mask_dataset', 'single', 'video_eval'], help="Mode: 'dataset', 'mask_dataset', 'single', or 'video_eval'")
    parser.add_argument('--model-type', type=str, default='yolo', help="Type of model adapter to use (e.g., 'yolo')")
    parser.add_argument('--model', type=str, required=True, help="Path to the model weights/file")
    parser.add_argument('--source', type=str, required=True, help="Path to dataset.yaml, image, or video file")
    parser.add_argument('--labels', type=str, default=None, help="Path to the labels directory (required for video_eval metrics, optional otherwise)")
    parser.add_argument('--task', type=str, default='seg', choices=['seg', 'det', 'ignore'], help="Task type: 'seg' (segmentation), 'det' (detection), or 'ignore' (global mask)")
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help="Dataset split to evaluate")
    parser.add_argument('--save-dir', type=str, default=None, help="Optional directory to save visual results (for single mode)")
    parser.add_argument('--iou-thresh', type=float, default=0.5, help="IoU threshold for considering a match")
    # Depth-Anything specific
    parser.add_argument('--depth-threshold', type=float, default=0.5, help="[depth model] Normalized threshold (0..1) for near/foreground detection")
    parser.add_argument('--depth-invert', action='store_true', help="[depth model] Select FAR objects instead of NEAR ones")
    
    args = parser.parse_args()
    
    # 1. Instantiate the appropriate Model Adapter
    model_adapter = get_model_adapter(
        args.model_type, args.model, args.task,
        _depth_threshold=args.depth_threshold,
        _depth_invert=args.depth_invert,
    )
    
    # 2. Pass it to the Generic Evaluator
    evaluator = GenericEvaluator(model_adapter=model_adapter, task=args.task)
    
    # 3. Run evaluation
    if args.mode == 'dataset':
        evaluator.evaluate_dataset(args.source, split=args.split, iou_threshold=args.iou_thresh)
    elif args.mode == 'mask_dataset':
        evaluator.evaluate_mask_dataset(args.source, iou_threshold=args.iou_thresh)
    elif args.mode == 'single':
        save_dir_abs = str(Path(args.save_dir).resolve()) if args.save_dir else None
        evaluator.evaluate_single(args.source, save_dir=save_dir_abs)
    elif args.mode == 'video_eval':
        evaluator.evaluate_video(args.source, args.labels, iou_threshold=args.iou_thresh)

if __name__ == "__main__":
    main()
