import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from data.data_loader import get_data_loaders
from model.UNet import UNet
from model.diffusion import Diffusion
from utils.visualization import plot_loss_curve

def train(model, diffusion, train_loader, optimizer, epochs, device, save_dir):
    """
    Train the diffusion model
    
    Args:
        model: The DDPM model
        diffusion: The diffusion process
        train_loader: DataLoader for training data
        optimizer: Optimizer
        epochs: Number of epochs to train for
        device: Device to train on
        save_dir: Directory to save checkpoints
        
    Returns:
        losses: List of average losses per epoch
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Track losses
    losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (images, conditions) in enumerate(progress_bar):
            images = images.to(device)
            conditions = conditions.to(device)
            
            # Sample random timesteps
            batch_size = images.shape[0]
            t = torch.randint(0, diffusion.noise_steps, (batch_size,), device=device).long() # low, high, size, 64 bits
            
            # Calculate loss
            loss = diffusion.p_losses(model, images, t, conditions)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average loss
        avg_loss = epoch_loss / len(train_loader)
        with open(os.path.join("./", "training_loss.txt"), "a") as f:
            f.write(f"{avg_loss:.7f}\n")
        # losses.append(avg_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Plot and save loss curve
    plot_loss_curve(losses, os.path.join(save_dir, "loss_curve.png"))
    
    return losses

def main(args):
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # Update config with command line arguments
    config = Config()
    config.BATCH_SIZE = args.batch_size if args.batch_size else config.BATCH_SIZE
    config.EPOCHS = args.epochs if args.epochs else config.EPOCHS
    config.LEARNING_RATE = args.lr if args.lr else config.LEARNING_RATE
    config.DEVICE = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    print(f"Using device: {config.DEVICE}")
    
    # Get data loaders
    train_loader, _, _, num_classes, label_to_idx = get_data_loaders(config)
    
    # Update condition dimension with actual number of classes
    config.CONDITION_DIM = num_classes
    
    # Initialize model
    model = UNet(
        in_channels=config.CHANNELS,
        out_channels=config.CHANNELS,
        time_dim=config.TIME_EMBEDDING_DIM,
        condition_dim=config.CONDITION_DIM
    ).to(config.DEVICE)
    
    # Initialize diffusion process
    diffusion = Diffusion(
        noise_steps=config.NOISE_STEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        img_size=config.IMAGE_SIZE,
        device=config.DEVICE
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Optionally load checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Train the model
    train(
        model,
        diffusion,
        train_loader,
        optimizer,
        config.EPOCHS - start_epoch,
        config.DEVICE,
        config.CHECKPOINT_DIR
    )
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional DDPM")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    main(args)