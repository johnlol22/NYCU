import torch
import os
import argparse
from tqdm import tqdm

from config import Config
from data.data_loader import get_data_loaders
from model.UNet import UNet
from model.diffusion import Diffusion
from utils.evaluation import evaluate_model
from utils.visualization import visualize_denoising_process

# Import the evaluator
from file.evaluator import evaluation_model

def main(args):
    """Main evaluation function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    _, test_loader, new_test_loader, num_classes, label_to_idx = get_data_loaders(Config)
    
    # Initialize model
    model = UNet(
        in_channels=Config.CHANNELS,
        out_channels=Config.CHANNELS,
        time_dim=Config.TIME_EMBEDDING_DIM,
        condition_dim=num_classes
    ).to(device)
    
    # Initialize diffusion process
    diffusion = Diffusion(
        noise_steps=Config.NOISE_STEPS,
        beta_start=Config.BETA_START,
        beta_end=Config.BETA_END,
        img_size=Config.IMAGE_SIZE,
        device=device
    )
    
    # Load checkpoint
    if not args.checkpoint:
        raise ValueError("Checkpoint must be provided for evaluation")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint['epoch']})")
    
    # Initialize evaluator
    evaluator = evaluation_model()          # import the given classifier
    # evaluator = evaluator.to(device)
    
    # Create results directories
    test_results_dir = os.path.join(Config.RESULTS_DIR, "test")
    new_test_results_dir = os.path.join(Config.RESULTS_DIR, "new_test")
    os.makedirs(test_results_dir, exist_ok=True)
    os.makedirs(new_test_results_dir, exist_ok=True)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_accuracy, test_images, _ = evaluate_model(         # call the evaluate interface to do inference
        model,
        diffusion,
        test_loader,
        evaluator,
        device=device,
        save_dir=test_results_dir
    )
    
    # Evaluate on new test set
    print("Evaluating on new test set...")
    new_test_accuracy, new_test_images, _ = evaluate_model(
        model,
        diffusion,
        new_test_loader,
        evaluator,
        device=device,
        save_dir=new_test_results_dir
    )
    
    # Create a file with results
    with open(os.path.join(Config.RESULTS_DIR, "evaluation_results.txt"), "w") as f:
        f.write(f"Test accuracy: {test_accuracy:.4f}\n")
        f.write(f"New test accuracy: {new_test_accuracy:.4f}\n")
    
    # Print results
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"New test accuracy: {new_test_accuracy:.4f}")
    
    # Create denoising process visualization for specific labels
    if args.visualize:
        print("Creating denoising process visualization...")
        
        # Set up the specific labels
        specific_labels = Config.SPECIFIC_LABELS
        
        # Visualize
        visualize_denoising_process(
            model,
            diffusion,
            None,  # No condition tensor, using labels instead
            label_to_idx=label_to_idx,
            labels=specific_labels,
            num_steps=Config.VIS_STEPS,
            save_path=os.path.join(Config.RESULTS_DIR, "denoising_process.png"),
            device=device
        )
        
        print(f"Denoising process visualization saved to {os.path.join(Config.RESULTS_DIR, 'denoising_process.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Conditional DDPM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Create denoising process visualization")
    
    args = parser.parse_args()
    main(args)