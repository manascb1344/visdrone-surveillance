import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class VisDroneDataset(Dataset):
    """
    PyTorch Dataset for VisDrone, compatible with torchvision Faster R-CNN.

    Loads images and annotations, formatting targets as required by Faster R-CNN:
    {'boxes': tensor([[x1, y1, x2, y2], ...]), 'labels': tensor([label_idx, ...])}
    where coordinates are unnormalized pixel values.
    """
    def __init__(self, images_dir, annotations_dir, transforms=None):
        """
        Args:
            images_dir (str): Directory containing the images.
            annotations_dir (str): Directory containing the VisDrone annotation files (.txt).
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

        # Mapping from original VisDrone class ID to our model's class ID
        # Original IDs: 1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van,
        #               6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor
        # We map these directly to 1-10, reserving 0 for background.
        # Original IDs 0 (ignored) and 11 (other) are skipped.
        self.visdrone_to_model_map = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
            6: 6, 7: 7, 8: 8, 9: 9, 10: 10
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # Corresponding annotation file
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(self.annotations_dir, f"{img_id}.txt")

        boxes = []
        labels = []

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue

                    # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
                    #                  <score>,<object_category>,<truncation>,<occlusion>
                    x_min = float(parts[0])
                    y_min = float(parts[1])
                    width = float(parts[2])
                    height = float(parts[3])
                    original_cls = int(parts[5])

                    # Skip ignored classes (0 and 11)
                    if original_cls not in self.visdrone_to_model_map:
                        continue

                    # Convert to [x_min, y_min, x_max, y_max]
                    x_max = x_min + width
                    y_max = y_min + height

                    # Get model class ID
                    model_cls = self.visdrone_to_model_map[original_cls]

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(model_cls)

        # Convert to tensors
        # Ensure boxes has shape [N, 4] even if N=0
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx]) # Add image_id, useful for evaluation
        # You might need to add area and iscrowd if required by evaluation scripts
        # target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        if self.transforms is not None:
            # Note: Transforms should handle both image and target updates (e.g., resizing boxes)
            # A simple ToTensor() won't work alone for object detection.
            # We'll need a more complex transform pipeline later.
            img, target = self.transforms(img, target)

        return img, target

def get_transform(train):
    """Defines basic transformations for the dataset."""
    import torchvision.transforms.v2 as T # Use v2 for joint image/target transforms
    transforms = []
    # Converts PIL image to tensor
    transforms.append(T.ToImageTensor())
    transforms.append(T.ConvertImageDtype(torch.float)) # Normalize to [0.0, 1.0]
    # Add data augmentation if training
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Add more augmentations like ColorJitter, RandomResizedCropWithBoxes etc.
        # Be careful: augmentations must also transform bounding boxes correctly.
        pass
    # Note: Resizing might be needed depending on the model or training strategy.
    # transforms.append(T.Resize((800, 800))) # Example resize

    return T.Compose(transforms)


# Example usage (for testing the dataset class)
if __name__ == "__main__":
    # !!! Important: Update these paths to your actual VisDrone dataset location !!!
    # These paths assume you have run the DataManager download/extract steps
    # and have the original VisDrone structure, NOT the YOLO converted one.
    base_data_path = "../visdrone" # Adjust if your data is elsewhere
    train_img_dir = os.path.join(base_data_path, "VisDrone2019-DET-train", "images")
    train_ann_dir = os.path.join(base_data_path, "VisDrone2019-DET-train", "annotations")

    val_img_dir = os.path.join(base_data_path, "VisDrone2019-DET-val", "images")
    val_ann_dir = os.path.join(base_data_path, "VisDrone2019-DET-val", "annotations")

    if not os.path.exists(train_img_dir) or not os.path.exists(train_ann_dir):
        print(f"Error: Training data not found at {train_img_dir} and {train_ann_dir}")
        print("Please ensure the VisDrone dataset is downloaded and extracted,")
        print("and update the base_data_path variable in this script.")
    else:
        print("Loading training dataset...")
        # Create dataset instance with basicToTensor transform for now
        train_dataset = VisDroneDataset(train_img_dir, train_ann_dir, transforms=get_transform(train=True))
        print(f"Training dataset size: {len(train_dataset)}")

        if len(train_dataset) > 0:
            # Get a sample
            img, target = train_dataset[0]
            print("\nSample Image Tensor:")
            print(f"  Shape: {img.shape}")
            print(f"  Type: {img.dtype}")
            print(f"  Min value: {img.min()}")
            print(f"  Max value: {img.max()}")

            print("\nSample Target Dictionary:")
            for key, value in target.items():
                print(f"  Key: '{key}'")
                if isinstance(value, torch.Tensor):
                    print(f"    Shape: {value.shape}")
                    print(f"    Dtype: {value.dtype}")
                    # Print first few elements if tensor is not empty
                    if value.numel() > 0:
                        print(f"    Value (first 5): {value.flatten()[:5]}...")
                    else:
                        print(f"    Value: (empty tensor)")
                else:
                    print(f"    Value: {value}")

            # Test dataloader with collate function
            # Torchvision detection models need a collate function that doesn't stack tensors,
            # but returns lists of images and targets.
            def collate_fn(batch):
                return tuple(zip(*batch))

            print("\nTesting DataLoader...")
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

            # Get one batch
            images, targets = next(iter(train_loader))
            print(f"Batch loaded. Number of images: {len(images)}")
            print(f"Type of images element: {type(images[0])}")
            print(f"Shape of first image tensor: {images[0].shape}")
            print(f"Number of targets: {len(targets)}")
            print(f"Type of targets element: {type(targets[0])}")
            print("Keys in first target dict:", targets[0].keys())

            print("\nDataset and DataLoader test finished.")
        else:
            print("Training dataset is empty. Cannot perform sample check.")

    # You can add similar checks for the validation set
    # val_dataset = VisDroneDataset(val_img_dir, val_ann_dir, transforms=get_transform(train=False))
    # print(f"Validation dataset size: {len(val_dataset)}") 