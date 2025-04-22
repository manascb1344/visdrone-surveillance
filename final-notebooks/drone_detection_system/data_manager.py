import os
import glob
import shutil
import yaml
import cv2
from tqdm import tqdm

class DataManager:
    def __init__(self, base_path="."):
        self.base_path = base_path
        self.visdrone_path = os.path.join(base_path, "visdrone")
        self.output_path = os.path.join(base_path, "visdrone_yolo")
        
        self.visdrone_to_yolo = {
            0: None,  # ignore
            1: 0,     # person
            2: 0,     # person
            3: 1,     # bicycle
            4: 2,     # car
            5: 3,     # van
            6: 4,     # truck
            7: 5,     # tricycle
            8: 5,     # awning-tricycle
            9: 6,     # bus
            10: 7,    # motor
            11: None  # ignore
        }
        
        self.class_names = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motorcycle']

    def download_visdrone(self):
        os.makedirs(self.visdrone_path, exist_ok=True)

        os.system(f"wget -c https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip -P {self.visdrone_path}/")
        os.system(f"wget -c https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip -P {self.visdrone_path}/")

        os.system(f"unzip -oq {self.visdrone_path}/VisDrone2019-DET-train.zip -d {self.visdrone_path}/")
        os.system(f"unzip -oq {self.visdrone_path}/VisDrone2019-DET-val.zip -d {self.visdrone_path}/")

        return self.visdrone_path

    def convert_visdrone_to_yolo(self):
        os.makedirs(f"{self.output_path}/images/train", exist_ok=True)
        os.makedirs(f"{self.output_path}/images/val", exist_ok=True)
        os.makedirs(f"{self.output_path}/labels/train", exist_ok=True)
        os.makedirs(f"{self.output_path}/labels/val", exist_ok=True)

        dataset_yaml = {
            'path': os.path.abspath(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        with open(f"{self.output_path}/dataset.yaml", 'w') as f:
            yaml.dump(dataset_yaml, f, sort_keys=False)

        print("Converting training set...")
        train_imgs = glob.glob(f"{self.visdrone_path}/VisDrone2019-DET-train/images/*.jpg")
        for img_path in tqdm(train_imgs):
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]

            shutil.copy(img_path, f"{self.output_path}/images/train/{img_name}")

            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            ann_path = f"{self.visdrone_path}/VisDrone2019-DET-train/annotations/{img_id}.txt"
            out_path = f"{self.output_path}/labels/train/{img_id}.txt"

            self._convert_annotation(ann_path, out_path, img_width, img_height)

        print("Converting validation set...")
        val_imgs = glob.glob(f"{self.visdrone_path}/VisDrone2019-DET-val/images/*.jpg")
        for img_path in tqdm(val_imgs):
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]

            shutil.copy(img_path, f"{self.output_path}/images/val/{img_name}")

            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            ann_path = f"{self.visdrone_path}/VisDrone2019-DET-val/annotations/{img_id}.txt"
            out_path = f"{self.output_path}/labels/val/{img_id}.txt"

            self._convert_annotation(ann_path, out_path, img_width, img_height)

        print(f"Conversion complete. Dataset stored in {self.output_path}")
        return f"{self.output_path}/dataset.yaml"

    def _convert_annotation(self, ann_path, out_path, img_width, img_height):
        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()

            with open(out_path, 'w') as f:
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue

                    x = int(parts[0])
                    y = int(parts[1])
                    width = int(parts[2])
                    height = int(parts[3])
                    cls = int(parts[5])

                    if cls not in self.visdrone_to_yolo or self.visdrone_to_yolo[cls] is None:
                        continue

                    yolo_x = (x + width / 2) / img_width
                    yolo_y = (y + height / 2) / img_height
                    yolo_width = width / img_width
                    yolo_height = height / img_height

                    yolo_x = min(max(yolo_x, 0.0), 1.0)
                    yolo_y = min(max(yolo_y, 0.0), 1.0)
                    yolo_width = min(max(yolo_width, 0.0), 1.0)
                    yolo_height = min(max(yolo_height, 0.0), 1.0)

                    yolo_class = self.visdrone_to_yolo[cls]
                    f.write(f"{yolo_class} {yolo_x:.6f} {yolo_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")
        except Exception as e:
            print(f"Error processing {ann_path}: {e}") 