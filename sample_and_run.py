import os
import random
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict

def main():
    # Шляхи
    dataset_dir = Path(r"c:\Users\Sasha\projects\CV\det_pipeline\yolo_det_dataset\images\val")
    output_input_dir = Path(r"c:\Users\Sasha\projects\CV\yolo_evaluator\results\100_frames_input")
    results_dir = Path(r"c:\Users\Sasha\projects\CV\yolo_evaluator\results")
    
    yolo_exe = r"C:\Users\Sasha\projects\CV\db_task\venv\Scripts\yolo.exe"
    det_model = r"c:\Users\Sasha\projects\CV\det_pipeline\runs\drone_det_n\weights\best.pt"
    seg_model = r"c:\Users\Sasha\projects\CV\db_task\models\best.pt"
    
    # 1. Збираємо і групуємо всі зображення
    if output_input_dir.exists():
        shutil.rmtree(output_input_dir)
    output_input_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(dataset_dir.glob("*.jpg"))
    video_groups = defaultdict(list)
    
    for img_path in images:
        stem = img_path.stem
        if "_f0" in stem:
            video_name = stem.rsplit("_f0", 1)[0]
        else:
            video_name = stem
        video_groups[video_name].append(img_path)
        
    num_videos = len(video_groups)
    print(f"Знайдено {len(images)} зображень з {num_videos} відео.")
    
    # 2. Вибираємо 100 кадрів (порівну з кожного відео)
    selected_images = []
    target_total = 100
    
    # Спочатку беремо порівну
    frames_per_video = target_total // num_videos
    leftover = target_total % num_videos
    
    videos = list(video_groups.keys())
    random.shuffle(videos)
    
    for i, vid in enumerate(videos):
        group_imgs = video_groups[vid]
        take = frames_per_video + (1 if i < leftover else 0)
        
        if len(group_imgs) >= take:
            selected_images.extend(random.sample(group_imgs, take))
        else:
            selected_images.extend(group_imgs) # Якщо кадрів менше ніж треба
            
    # Якщо вийшло менше 100 через брак кадрів у деяких відео, добираємо рандомні з інших
    if len(selected_images) < target_total:
        remaining = set(images) - set(selected_images)
        needed = target_total - len(selected_images)
        selected_images.extend(random.sample(list(remaining), min(needed, len(remaining))))
        
    print(f"Відібрано {len(selected_images)} кадрів. Копіювання...")
    for img in selected_images:
        shutil.copy(img, output_input_dir / img.name)
        
    # 3. Запуск YOLO для детекції
    print("\n🚀 Запуск детекції (інтерфейс)...")
    subprocess.run([
        yolo_exe, "predict",
        f"model={det_model}",
        f"source={output_input_dir}",
        "save=True",
        f"project={results_dir}",
        "name=det_100_frames",
        "exist_ok=True"
    ], check=True)
    
    # 4. Запуск YOLO для сегментації
    print("\n🚀 Запуск сегментації (горизонт)...")
    subprocess.run([
        yolo_exe, "predict",
        f"model={seg_model}",
        f"source={output_input_dir}",
        "save=True",
        f"project={results_dir}",
        "name=seg_100_frames",
        "exist_ok=True"
    ], check=True)
    
    print("\n✅ Готово! Результати збережено у:")
    print(f" - Детекція: {results_dir / 'det_100_frames'}")
    print(f" - Сегментація: {results_dir / 'seg_100_frames'}")

if __name__ == "__main__":
    main()
