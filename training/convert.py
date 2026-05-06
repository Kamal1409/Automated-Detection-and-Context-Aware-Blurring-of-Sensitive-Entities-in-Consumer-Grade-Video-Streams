# This is a function that converts the widerface dataset annotation to a YOLO format.

import os
import cv2
import shutil

BASE_DIR = os.path.join("training", "data", "widerface")
OUTPUT_DIR = os.path.join("training", "faces", "dataset")


def convert(split):
    if split == "train":
        ann_file = os.path.join(
            BASE_DIR, "wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt"
        )
        img_root = os.path.join(BASE_DIR, "WIDER_train/WIDER_train/images")
    else:
        ann_file = os.path.join(
            BASE_DIR, "wider_face_split/wider_face_split/wider_face_val_bbx_gt.txt"
        )
        img_root = os.path.join(BASE_DIR, "WIDER_val/WIDER_val/images")

    out_img_dir = os.path.join(OUTPUT_DIR, f"images/{split}")
    out_lbl_dir = os.path.join(OUTPUT_DIR, f"labels/{split}")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    with open(ann_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        img_rel_path = lines[i].strip()
        i += 1

        num_boxes = int(lines[i].strip())
        i += 1

        img_path = os.path.join(img_root, img_rel_path)

        if not os.path.exists(img_path):
            i += num_boxes
            continue

        # Copy image once
        new_img_path = os.path.join(out_img_dir, os.path.basename(img_rel_path))
        if not os.path.exists(new_img_path):
            shutil.copy(img_path, new_img_path)

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        label_path = os.path.join(
            out_lbl_dir, os.path.basename(img_rel_path).replace(".jpg", ".txt")
        )

        if os.path.exists(label_path):
            continue

        with open(label_path, "w") as out:
            for _ in range(num_boxes):
                parts = list(map(int, lines[i].strip().split()))
                i += 1

                x, y, bw, bh = parts[:4]

                # Convert to YOLO format
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h

                out.write(f"0 {xc} {yc} {bw} {bh}\n")

        if i % 1000 == 0:
            print(f"Processed {i} lines...")


# Run conversion
convert("train")
convert("val")

print("✅ Conversion completed!")
