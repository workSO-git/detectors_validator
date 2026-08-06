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
    parser = argparse.ArgumentParser(description="Generic Model Evaluator (mIoU, Precision, Recall)")
    parser.add_argument('--mode', type=str, required=True, choices=['dataset', 'single'], help="Mode: 'dataset' or 'single'")
    parser.add_argument('--model-type', type=str, default='yolo', help="Type of model adapter to use (e.g., 'yolo')")
    parser.add_argument('--model', type=str, required=True, help="Path to the model weights/file")
    parser.add_argument('--source', type=str, required=True, help="Path to dataset.yaml or image/video file")
    parser.add_argument('--task', type=str, default='seg', choices=['seg', 'det'], help="Task type: 'seg' (segmentation) or 'det' (detection)")
    parser.add_argument('--save-dir', type=str, default=None, help="Optional directory to save visual results (for single mode)")
    parser.add_argument('--iou-thresh', type=float, default=0.5, help="IoU threshold for considering a match (dataset mode)")
    
    args = parser.parse_args()
    
    # 1. Instantiate the appropriate Model Adapter
    model_adapter = get_model_adapter(args.model_type, args.model, args.task)
    
    # 2. Pass it to the Generic Evaluator
    evaluator = GenericEvaluator(model_adapter=model_adapter, task=args.task)
    
    # 3. Run evaluation
    if args.mode == 'dataset':
        evaluator.evaluate_dataset(args.source, iou_threshold=args.iou_thresh)
    elif args.mode == 'single':
        evaluator.evaluate_single(args.source, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
