import os
import random
import yaml
from ultralytics import YOLO

if __name__ == '__main__':
    print("Step 1: Auto-discovering dataset and rebuilding data.yaml...")
    img_dir, lbl_dir = None, None
    
    # Auto-find the dataset
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'images' in dirs and img_dir is None:
            img_dir = os.path.join(root, 'images')
        if 'labels' in dirs and lbl_dir is None:
            lbl_dir = os.path.join(root, 'labels')

    if not img_dir or not lbl_dir:
        raise Exception("Dataset not found! Make sure WIDER FACE is attached in the right panel.")

    # Match images to labels
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    valid_imgs = []
    for img in all_imgs:
        if os.path.exists(os.path.join(lbl_dir, os.path.splitext(img)[0] + '.txt')):
            valid_imgs.append(os.path.join(img_dir, img))

    # Rebuild the Train/Val split
    random.seed(42)
    random.shuffle(valid_imgs)
    split_idx = int(len(valid_imgs) * 0.8)

    with open('/kaggle/working/train.txt', 'w') as f:
        f.write('\n'.join(valid_imgs[:split_idx]))
    with open('/kaggle/working/val.txt', 'w') as f:
        f.write('\n'.join(valid_imgs[split_idx:]))

    # Rebuild the yaml file
    yaml_content = {
        'train': '/kaggle/working/train.txt', 
        'val': '/kaggle/working/val.txt',
        'nc': 1,
        'names': ['face']
    }
    with open('/kaggle/working/data.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

    print("Step 2: data.yaml rebuilt successfully! Booting Dual-GPU YOLO...")
    
    model = YOLO('yolov8n.pt')
    model.train(
        data='/kaggle/working/data.yaml',
        project='/kaggle/working/runs',
        name='spixgro_face_detector',
        
        # --- DUAL GPU SETUP ---
        device=[0, 1],     
        batch=16,          # Safe batch size to prevent OOM on crowd scenes
        workers=2,         
        
        # --- Core Params ---
        epochs=100,
        patience=20,
        imgsz=640,
        
        save=True,
        save_period=10,
        plots=True,
        val=True,
        
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
