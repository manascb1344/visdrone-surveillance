import os
import glob
import shutil
import yaml
import cv2
from tqdm import tqdm

class DataManager:
    """
    Class for managing the VisDrone dataset, including downloading, 
    extracting, and converting to YOLO format.
    """
    def __init__(self, base_path="."):
        """
        Initialize DataManager with base path for dataset storage
        
        Args:
            base_path (str): Base path for storing datasets
        """
        self.base_path = base_path
        self.visdrone_path = os.path.join(base_path, "visdrone")
        self.output_path = os.path.join(base_path, "visdrone_yolo")
        
        # Define VisDrone to YOLO class mapping
        # We'll map similar classes and ignore some
        self.visdrone_to_yolo = {
            0: None,  # ignored regions -> ignore
            1: 0,     # pedestrian -> person
            2: 0,     # person -> person
            3: 1,     # bicycle -> bicycle
            4: 2,     # car -> car
            5: 3,     # van -> van
            6: 4,     # truck -> truck
            7: 5,     # tricycle -> custom
            8: 5,     # awning-tricycle -> custom
            9: 6,     # bus -> bus
            10: 7,    # motor -> motorcycle
            11: None  # others -> ignore
        }
        
        # Define class names
        self.class_names = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motorcycle']

    def download_visdrone(self):
        """
        Download and extract the VisDrone dataset
        """
        print("Downloading VisDrone dataset...")
        os.makedirs(self.visdrone_path, exist_ok=True)

        # Download the dataset files
        # Note: In a production environment, replace these system calls with Python requests or urllib
        os.system(f"wget -c https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip -P {self.visdrone_path}/")
        os.system(f"wget -c https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip -P {self.visdrone_path}/")

        print("Extracting dataset files...")
        os.system(f"unzip -oq {self.visdrone_path}/VisDrone2019-DET-train.zip -d {self.visdrone_path}/")
        os.system(f"unzip -oq {self.visdrone_path}/VisDrone2019-DET-val.zip -d {self.visdrone_path}/")

        print("VisDrone dataset downloaded and extracted.")
        return self.visdrone_path

    def convert_visdrone_to_yolo(self):
        """
        Convert VisDrone annotations to YOLO format

        VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized coordinates)
        
        Returns:
            str: Path to the generated dataset.yaml file
        """
        # Create output directories
        os.makedirs(f"{self.output_path}/images/train", exist_ok=True)
        os.makedirs(f"{self.output_path}/images/val", exist_ok=True)
        os.makedirs(f"{self.output_path}/labels/train", exist_ok=True)
        os.makedirs(f"{self.output_path}/labels/val", exist_ok=True)

        # Create dataset.yaml file
        dataset_yaml = {
            'path': os.path.abspath(self.output_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        with open(f"{self.output_path}/dataset.yaml", 'w') as f:
            yaml.dump(dataset_yaml, f, sort_keys=False)

        # Process training set
        print("Converting training set...")
        train_imgs = glob.glob(f"{self.visdrone_path}/VisDrone2019-DET-train/images/*.jpg")
        for img_path in tqdm(train_imgs):
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]

            # Copy image to output directory
            shutil.copy(img_path, f"{self.output_path}/images/train/{img_name}")

            # Get image dimensions
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            # Convert annotation
            ann_path = f"{self.visdrone_path}/VisDrone2019-DET-train/annotations/{img_id}.txt"
            out_path = f"{self.output_path}/labels/train/{img_id}.txt"

            self._convert_annotation(ann_path, out_path, img_width, img_height)

        # Process validation set
        print("Converting validation set...")
        val_imgs = glob.glob(f"{self.visdrone_path}/VisDrone2019-DET-val/images/*.jpg")
        for img_path in tqdm(val_imgs):
            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]

            # Copy image to output directory
            shutil.copy(img_path, f"{self.output_path}/images/val/{img_name}")

            # Get image dimensions
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            # Convert annotation
            ann_path = f"{self.visdrone_path}/VisDrone2019-DET-val/annotations/{img_id}.txt"
            out_path = f"{self.output_path}/labels/val/{img_id}.txt"

            self._convert_annotation(ann_path, out_path, img_width, img_height)

        print(f"Conversion complete. Dataset stored in {self.output_path}")
        return f"{self.output_path}/dataset.yaml"

    def _convert_annotation(self, ann_path, out_path, img_width, img_height):
        """
        Convert a single annotation file from VisDrone to YOLO format
        
        Args:
            ann_path (str): Path to the input annotation file
            out_path (str): Path to the output YOLO format annotation file
            img_width (int): Width of the image
            img_height (int): Height of the image
        """
        try:
            with open(ann_path, 'r') as f:
                lines = f.readlines()

            with open(out_path, 'w') as f:
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue

                    # Get bounding box coordinates and class
                    x = int(parts[0])
                    y = int(parts[1])
                    width = int(parts[2])
                    height = int(parts[3])
                    cls = int(parts[5])

                    # Skip if class is ignored
                    if cls not in self.visdrone_to_yolo or self.visdrone_to_yolo[cls] is None:
                        continue

                    # Convert to YOLO format: <class> <x_center> <y_center> <width> <height>
                    yolo_x = (x + width / 2) / img_width
                    yolo_y = (y + height / 2) / img_height
                    yolo_width = width / img_width
                    yolo_height = height / img_height

                    # Ensure values are within bounds
                    yolo_x = min(max(yolo_x, 0.0), 1.0)
                    yolo_y = min(max(yolo_y, 0.0), 1.0)
                    yolo_width = min(max(yolo_width, 0.0), 1.0)
                    yolo_height = min(max(yolo_height, 0.0), 1.0)

                    # Write to output file
                    yolo_class = self.visdrone_to_yolo[cls]
                    f.write(f"{yolo_class} {yolo_x:.6f} {yolo_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")
        except Exception as e:
            print(f"Error processing {ann_path}: {e}") 