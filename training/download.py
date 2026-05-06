import os
import shutil
import kagglehub

base_path = "D:/Projects/GDSH/SDP/training/data"

datasets = {
    "widerface": "mksaad/wider-face-a-face-detection-benchmark",
    "casia": "debarghamitraroy/casia-webface",
    "vggface2": "hearfool/vggface2",
}

for name, ds in datasets.items():
    temp_path = kagglehub.dataset_download(ds)
    target = os.path.join(base_path, name)
    os.makedirs(target, exist_ok=True)

    for name_or_file in os.listdir(temp_path):
        src = os.path.join(temp_path, name_or_file)
        dst = os.path.join(target, name_or_file)

        if os.path.exists(dst):
            continue

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
