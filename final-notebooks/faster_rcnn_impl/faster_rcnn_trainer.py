import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

# Import from our other modules
from faster_rcnn_dataset import VisDroneDataset, get_transform
from faster_rcnn_model import get_faster_rcnn_model

# --- Configuration ---
# Adjust these paths and parameters as needed
base_data_path = "../visdrone"  # Path to the extracted VisDrone dataset (parent dir of VisDrone2019-DET-train)
output_dir = "./faster_rcnn_output" # Directory to save checkpoints
os.makedirs(output_dir, exist_ok=True)

num_classes = 11  # 10 VisDrone object classes + 1 background
learning_rate = 0.005
batch_size = 4 # Adjust based on GPU memory
num_epochs = 10 # Number of training epochs
weight_decay = 0.0005
momentum = 0.9
lr_step_size = 3
lr_gamma = 0.1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# --- Helper Functions ---
def collate_fn(batch):
    """Custom collate function for object detection dataloader."""
    return tuple(zip(*batch))

# --- Data Loading ---
print("Setting up datasets and dataloaders...")

# Define dataset paths (using original VisDrone structure)
train_img_dir = os.path.join(base_data_path, "VisDrone2019-DET-train", "images")
train_ann_dir = os.path.join(base_data_path, "VisDrone2019-DET-train", "annotations")
val_img_dir = os.path.join(base_data_path, "VisDrone2019-DET-val", "images")
val_ann_dir = os.path.join(base_data_path, "VisDrone2019-DET-val", "annotations")

# Check if data paths exist
if not os.path.exists(train_img_dir) or not os.path.exists(train_ann_dir) or \
   not os.path.exists(val_img_dir) or not os.path.exists(val_ann_dir):
    print("Error: VisDrone dataset directories not found at expected locations.")
    print(f"Checked base path: {base_data_path}")
    print("Please ensure the dataset is downloaded/extracted and base_data_path is correct.")
    exit()

# Create datasets
train_dataset = VisDroneDataset(train_img_dir, train_ann_dir, transforms=get_transform(train=True))
val_dataset = VisDroneDataset(val_img_dir, val_ann_dir, transforms=get_transform(train=False))

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4, # Adjust based on your system
    collate_fn=collate_fn,
    pin_memory=True # Helps speed up data transfer to GPU
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size, # Often use smaller batch size for validation if needed
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

# --- Model Initialization ---
print("Initializing model...")
model = get_faster_rcnn_model(num_classes=num_classes)
model.to(device)

# --- Optimizer and Scheduler ---
print("Setting up optimizer and scheduler...")
# Gather parameters that require gradients
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Learning rate scheduler
lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

# --- Training Loop ---
print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for i, (images, targets) in enumerate(progress_bar):
        # Move data to the correct device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass - model returns loss dictionary in training mode
        loss_dict = model(images, targets)

        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())

        # Check for invalid loss
        if not torch.isfinite(losses):
            print(f"\nWarning: Non-finite loss detected: {losses.item()}. Skipping batch {i}.")
            print("Loss dict:", loss_dict)
            # Consider logging more info or stopping training if this persists
            continue

        # Backward pass
        losses.backward()

        # Optimizer step
        optimizer.step()

        epoch_loss += losses.item()

        # Update progress bar
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)

    # Update the learning rate
    lr_scheduler.step()

    print(f"\nEpoch {epoch+1}/{num_epochs} finished.")
    print(f"  Average Training Loss: {avg_epoch_loss:.4f}")
    print(f"  Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # --- Optional: Basic Validation (Loss calculation) ---
    # More comprehensive validation (e.g., mAP) requires evaluation libraries.
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad(): # Disable gradient calculations
        val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for images, targets in val_progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Note: During evaluation, the model needs targets to compute loss internally
            # If you only want predictions, you'd call model(images) without targets.
            # For simplicity, we calculate loss here like in training.
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isfinite(losses):
                val_loss += losses.item()
            else:
                 print(f"\nWarning: Non-finite validation loss detected: {losses.item()}")

    avg_val_loss = val_loss / len(val_loader)
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")

    # Save a checkpoint
    checkpoint_path = os.path.join(output_dir, f'fasterrcnn_visdrone_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': avg_epoch_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTraining finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Final model checkpoint saved in {output_dir}") 