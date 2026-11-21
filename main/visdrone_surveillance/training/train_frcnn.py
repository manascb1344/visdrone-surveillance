import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse

from .. import config
from ..data.visdrone_dataset import create_data_loaders
from ..models.faster_rcnn import get_model, save_model
from ..utils.evaluation import evaluate_model
from ..utils.visualization import visualize_predictions, plot_metrics

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        data_loader: Training data loader
        device: Device to train on (CPU or GPU)
        epoch: Current epoch number
    
    Returns:
        float: Average loss for this epoch
    """
    # Set model to training mode
    model.train()
    
    # Initialize metrics
    epoch_loss = 0
    
    # Create progress bar
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    # Train for one epoch
    for images, targets in progress_bar:
        # Move images and targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
        
        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss_value
    
    # Calculate average loss
    avg_loss = epoch_loss / len(data_loader)
    
    return avg_loss


def train_model(args):
    """
    Train the object detection model
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(batch_size=args.batch_size)
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    # Get model
    model = get_model(num_classes=len(config.CLASSES))
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Create output directory for checkpoints and visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")
    
    # Initialize tracking variables
    best_map = 0.0
    train_losses = []
    val_maps = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Train for one epoch
        epoch_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_losses.append(epoch_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        print(f"Evaluating on validation set...")
        metrics = evaluate_model(model, val_loader, device)
        val_maps.append(metrics['mAP'])
        
        # Print metrics
        print(f"Epoch {epoch} results:")
        print(f"  Training loss: {epoch_loss:.4f}")
        print(f"  Validation mAP: {metrics['mAP']:.4f}")
        
        # Save best model
        if metrics['mAP'] > best_map:
            best_map = metrics['mAP']
            print(f"  New best mAP: {best_map:.4f}, saving model...")
            
            # Save model checkpoint
            model_path = os.path.join(output_dir, f"best_model.pth")
            save_model(model, model_path)
        
        # Periodically save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            # Save model checkpoint
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
            save_model(model, checkpoint_path)
            
            # Visualize sample predictions
            print("Generating visualizations...")
            sample_data = next(iter(val_loader))
            images, targets = sample_data
            
            # Visualize predictions
            fig = visualize_predictions(
                model, images, targets, device, 
                class_names=[config.CLASSES[i] for i in range(len(config.CLASSES))],
                threshold=0.5
            )
            
            # Save visualization
            fig.savefig(os.path.join(output_dir, f"predictions_epoch_{epoch}.png"))
            plt.close(fig)
            
            # Plot and save metrics
            metrics_fig = plot_metrics(metrics)
            metrics_fig.savefig(os.path.join(output_dir, f"metrics_epoch_{epoch}.png"))
            plt.close(metrics_fig)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_maps)
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    
    print(f"Training completed. Best mAP: {best_map:.4f}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train object detection model on VisDrone dataset")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Train model
    train_model(args) 