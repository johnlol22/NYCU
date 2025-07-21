import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from oxford_pet import load_dataset

# Import the UNet model
from models import unet, resnet34_unet, resnet34_unet_modify
from evaluate import * 

def load_model(model_path, device):
    """
    Load a trained UNet model from a checkpoint file
    
    Args:
        model_path (str): Path to the model checkpoint file
        device (torch.device): Device to load the model on (cpu or cuda)
        
    Returns:
        model (UNet): Loaded model ready for inference
    """
    # model for UNet
    model = unet.UNet(n_channels=3, n_classes=1)
    # model for common ResUNet
    # model = resnet34_unet.ResUNet()
    # model for ResUNet on paper
    # model = resnet34_unet_modify.ResUNet()


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Path to the trained model
    model_path = args.model
    
    # Load the model
    model = load_model(model_path, device)
    test_dataset = load_dataset(args.data_path, mode='test')


    dice = evaluate(model, test_dataset, device)
    print(f'Dice score: {dice}')
    
if __name__ == '__main__':
    args = get_args()
    main()