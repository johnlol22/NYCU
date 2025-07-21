import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from oxford_pet import SimpleOxfordPetDataset, load_dataset
from oxford_pet import TqdmUpTo as tqdm
from models import unet, resnet34_unet, resnet34_unet_modify
import matplotlib.pyplot as plt
from evaluate import evaluate
from utils import *

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = load_dataset(args.data_path, mode='train')
    val_dataset = load_dataset(args.data_path, mode='valid')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,                   # random the train set
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    # model for unet
    model = unet.UNet(n_channels=3, n_classes=1)
    # model for common ResUNet
    # model = resnet34_unet.ResUNet()
    # model for ResUNet on paper
    # model = resnet34_unet_modify.ResUNet()
    model.to(device)
    
    # Define loss function and optimizer
    # auto weight decay
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    

    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=20,    # Reduce LR every 15 epochs
    #     gamma=0.5        # Multiply LR by 0.5 at each step
    # )
    
    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch') as pbar:
            for batch in train_loader:
                # Get data
                images = torch.tensor(np.stack(batch['image']), dtype=torch.float32).to(device)
                true_masks = torch.tensor(np.stack(batch['mask']), dtype=torch.float32).to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                masks_pred = model(images)
                
                # Calculate Dice loss
                loss = criterion(masks_pred, true_masks)
                # loss = dice_loss(masks_pred, true_masks)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                
                # Update progress
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        
        # Calculate average epoch loss
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = torch.tensor(np.stack(batch['image']), dtype=torch.float32).to(device)
                true_masks = torch.tensor(np.stack(batch['mask']), dtype=torch.float32).to(device)
                
                # Forward pass
                masks_pred = model(images)
                
                # Calculate loss
                loss = criterion(masks_pred, true_masks)
                # loss = dice_loss(masks_pred, true_masks)
                val_loss += loss.item()
        val_dice = evaluate(model, val_dataset, device)
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)


        # Update learning rate
        # scheduler.step()
        
        # Print epoch stats
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice Score: {val_dice:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./saved_models/output_{epoch}_{val_loss}.pth')
            print(f'Checkpoint saved! Val Loss: {val_loss:.4f}')
    
    plot(train_losses, val_losses)
    
    print('Training completed!')
    return model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)