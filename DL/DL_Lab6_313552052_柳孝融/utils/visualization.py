import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

def visualize_denoising_process(model, diffusion, condition, label_to_idx=None, labels=None, 
                              num_steps=8, save_path="denoising_process.png", device="cuda"):
    """
    Visualize the denoising process for a specific condition
    
    Args:
        model: Trained DDPM model
        diffusion: Diffusion process
        condition: Condition tensor or list of labels
        label_to_idx: Mapping from label strings to indices (needed if labels provided)
        labels: List of label strings (alternative to condition tensor)
        num_steps: Number of steps to visualize
        save_path: Path to save the visualization
        device: Device to run on
    """
    model.eval()
    
    # Convert labels to condition tensor if needed
    if labels is not None and label_to_idx is not None:
        num_classes = len(label_to_idx)
        condition = torch.zeros(1, num_classes, device=device)  # return size: 1*num_classes
        for label in labels:
            if label in label_to_idx:  # Check if label exists in mapping
                condition[0, label_to_idx[label]] = 1.0
            else:
                print(f"Warning: Label '{label}' not found in label dictionary.")
    elif condition is not None:
        # Ensure condition is on the correct device
        condition = condition.to(device)
    else:
        raise ValueError("Either condition tensor or labels must be provided")
    
    # Start from pure noise
    img_size = diffusion.img_size
    img = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Determine steps to visualize
    step_size = diffusion.noise_steps // num_steps
    steps_to_visualize = list(range(diffusion.noise_steps-1, -1, -step_size))
    steps_to_visualize.reverse()  # Reverse to go from noisy to clean, for showing result purpose
    
    # Add initial noise
    images = [img.cpu().clone()]
    
    # Sampling loop
    for i in tqdm(reversed(range(diffusion.noise_steps)), desc="Visualizing denoising"):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        with torch.no_grad():
            img = diffusion.p_sample(model, img, t, condition, i)
            
            # Save image at selected timesteps
            if i in steps_to_visualize:
                images.append(img.cpu().clone())
    
    # Make sure we have the final denoised image
    if 0 not in steps_to_visualize:
        # Get the last (fully denoised) image
        t_final = torch.zeros(1, device=device, dtype=torch.long)
        with torch.no_grad():
            final_img = diffusion.p_sample(model, img, t_final, condition, 0)
            images.append(final_img.cpu().clone())
    
    # Process images for display
    processed_images = []
    for img in images:
        # Convert from [-1, 1] to [0, 1]
        norm_img = (img + 1) / 2
        norm_img = torch.clamp(norm_img, 0, 1)
        processed_images.append(norm_img)
    
    # Create grid
    all_images = torch.cat(processed_images, dim=0)
    grid = make_grid(all_images, nrow=len(processed_images), padding=2)
    
    # Save as image
    save_image(grid, save_path)
    
    # Also create a matplotlib figure for better visualization
    plt.figure(figsize=(20, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    
    # Add title if labels were provided
    if labels is not None:
        plt.title(f"Denoising Process: {', '.join(labels)}")
    
    plt.tight_layout()      # adjust the distance for every subimage
    plt.savefig(save_path.replace('.png', '_plot.png'), bbox_inches='tight')
    plt.close()
    
    return grid



def plot_loss_curve(losses, save_path=None):
    """
    Plot the training loss curve
    
    Args:
        losses: List of loss values
        save_path: Path to save the plot (or None to just display)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()