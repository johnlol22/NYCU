import matplotlib.pyplot as plt
import numpy as np
import torch
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    """
    Calculate Dice coefficient for evaluation
    
    Args:
        pred_mask (numpy.ndarray): Predicted binary mask
        true_mask (numpy.ndarray): Ground truth binary mask
        
    Returns:
        float: Dice coefficient
    """
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.detach().cpu().numpy()
    intersection = np.sum(pred_mask * gt_mask)
    dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask))
    
    return dice

def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice loss for training
    """
    # Apply sigmoid to logits to get probabilities
    pred = torch.sigmoid(pred)
    
    # Flatten tensors
    batch_size = pred.size(0)
    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)
    
    # Make sure tensors have the right dtype and are on the same device
    pred = pred.float()
    target = target.float()
    
    # Calculate Dice coefficient (maintaining gradient information)
    intersection = torch.sum(pred * target, dim=1)
    pred_sum = torch.sum(pred, dim=1)
    target_sum = torch.sum(target, dim=1)
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Return loss (1 - Dice)
    return 1.0 - dice.mean()

def plot(train_losses, val_losses):
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    
    return

