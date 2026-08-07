import argparse
from pathlib import Path
from evaluator import GenericEvaluator

def get_model_adapter(model_type, model_path, task):
    """Factory function to instantiate the correct model adapter."""
    if model_type == 'yolo':
        from models.yolo_model import YoloModel
        return YoloModel(model_path, task=task)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description="Generic Model Evaluator (mIoU, Precision, Recall, F1, Jitter, Flicker)")
    parser.add_argument('--mode', type=str, required=True, choices=['dataset', 'single', 'video_eval'], help="Mode: 'dataset', 'single', or 'video_eval'")
    parser.add_argument('--model-type', type=str, default='yolo', help="Type of model adapter to use (e.g., 'yolo')")
    parser.add_argument('--model', type=str, required=True, help="Path to the model weights/file")
    parser.add_argument('--source', type=str, required=True, help="Path to dataset.yaml, image, or video file")
    parser.add_argument('--labels', type=str, default=None, help="Path to the labels directory (required for video_eval metrics, optional otherwise)")
    parser.add_argument('--task', type=str, default='seg', choices=['seg', 'det'], help="Task type: 'seg' (segmentation) or 'det' (detection)")
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help="Dataset split to evaluate")
    parser.add_argument('--save-dir', type=str, default=None, help="Optional directory to save visual results (for single mode)")
    parser.add_argument('--iou-thresh', type=float, default=0.5, help="IoU threshold for considering a match")
    
    args = parser.parse_args()
    
    # 1. Instantiate the appropriate Model Adapter
    model_adapter = get_model_adapter(args.model_type, args.model, args.task)
    
    # 2. Pass it to the Generic Evaluator
    evaluator = GenericEvaluator(model_adapter=model_adapter, task=args.task)
    
    # 3. Run evaluation
    if args.mode == 'dataset':
        evaluator.evaluate_dataset(args.source, split=args.split, iou_threshold=args.iou_thresh)
    elif args.mode == 'single':
        save_dir_abs = str(Path(args.save_dir).resolve()) if args.save_dir else None
        evaluator.evaluate_single(args.source, save_dir=save_dir_abs)
    elif args.mode == 'video_eval':
        evaluator.evaluate_video(args.source, args.labels, iou_threshold=args.iou_thresh)

if __name__ == "__main__":
    main()
