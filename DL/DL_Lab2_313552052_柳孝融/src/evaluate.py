import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *

def evaluate(model, dataset, device):
    """
    Evaluate the model on the Oxford-IIIT Pet dataset
    
    Args:
        model (UNet): Trained UNet model
        dataset (SimpleOxfordPetDataset): Dataset containing test images and ground truth masks
        device (torch.device): Device to perform inference on
        
    Returns:
        dict: Dictionary containing average metrics across the dataset
    """
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize metrics
    total_dice = 0.0
    
    # Set model to evaluation mode
    model.eval()
    
    # Process each image
    with torch.no_grad():
        for sample in dataloader:
            # Get image and ground truth mask
            image = torch.tensor(np.stack(sample['image']), dtype=torch.float32).to(device)
            true_mask = torch.tensor(np.stack(sample['mask']), dtype=torch.float32).to(device)

            # Predict mask
            output = model(image)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.float32)

            # Calculate metrics
            dice = dice_score(pred_mask, true_mask)
            total_dice += dice
    
    # Calculate average metrics
    num_samples = len(dataset)
    avg_dice = total_dice/num_samples
    
    return avg_dice