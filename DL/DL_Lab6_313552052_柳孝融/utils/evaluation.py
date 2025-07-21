import torch
import torch.nn as nn
import os
from torchvision import transforms
from tqdm import tqdm

class EvaluatorInterface:
    """Interface for working with the provided evaluator"""
    
    def __init__(self, evaluator):
        """
        Initialize with the provided evaluator
        
        Args:
            evaluator: The pretrained evaluator model
        """
        self.evaluator = evaluator
        self.normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    def evaluate_images(self, images, labels, batch_size=32, device="cuda"):
        """
        Evaluate generated images using the provided evaluator
        
        Args:
            images: Tensor of images in [0, 1] range
            labels: Tensor of one-hot encoded labels
            batch_size: Batch size for evaluation
            device: Device to run on
            
        Returns:
            accuracy: The accuracy score
        """
        # Process in batches to avoid memory issues
        total_acc = 0
        total_samples = 0
        
        for i in range(0, len(images), batch_size):
            # Get batch
            batch_images = images[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)
            
            # Normalize for evaluator
            normalized_images = self.normalizer(batch_images)
            
            # Evaluate
            batch_acc = self.evaluator.eval(normalized_images, batch_labels)
            batch_size_actual = batch_images.size(0)
            total_acc += batch_acc * batch_size_actual
            total_samples += batch_size_actual
        
        # Calculate overall accuracy
        accuracy = total_acc / total_samples
        
        return accuracy

def evaluate_model(model, diffusion, test_loader, evaluator, device="cuda", save_dir=None):
    """
    Evaluate the model on a test dataset
    
    Args:
        model: The trained DDPM model
        diffusion: The diffusion process
        test_loader: DataLoader for the test dataset
        evaluator: The evaluator model
        device: Device to run on
        save_dir: Directory to save results (or None to skip saving)
        
    Returns:
        accuracy: The accuracy score
        generated_images: The generated images
        filenames: The filenames corresponding to the generated images
    """
    model.eval()
    evaluator_interface = EvaluatorInterface(evaluator)                 # define the inference method in the interface
    
    # Create save directory if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    all_images = []
    all_labels = []
    all_filenames = []
    
    # Generate images for each batch
    for labels, filenames in tqdm(test_loader, desc="Generating images"):
        labels = labels.to(device)
        
        # Generate images
        with torch.no_grad():
            generated_images = diffusion.sample(model, labels)
            
            # Convert from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1) / 2
            generated_images = torch.clamp(generated_images, 0, 1)
        
        # Store results
        all_images.append(generated_images.cpu())
        all_labels.append(labels.cpu())
        all_filenames.extend(filenames)
    
    # Concatenate results
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Evaluate
    accuracy = evaluator_interface.evaluate_images(all_images, all_labels, device=device)
    
    # Save results if requested
    if save_dir is not None:
        # Save individual images
        for i, filename in enumerate(all_filenames):
            # Extract the base filename without extension
            base_filename = os.path.splitext(os.path.basename(filename))[0]             # just the number
            # torch.save(all_images[i], os.path.join(save_dir, f"{base_filename}.pt"))
            
            # Also save as PNG
            from torchvision.utils import save_image
            save_image(all_images[i], os.path.join(save_dir, f"{base_filename}.png"))
        
        # Create grid for visualization
        from torchvision.utils import make_grid
        grid = make_grid(all_images[:min(32, len(all_images))], nrow=8)
        save_image(grid, os.path.join(save_dir, "grid.png"))
        
        # Save accuracy
        with open(os.path.join(save_dir, "accuracy.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy:.6f}\n")
    
    return accuracy, all_images, all_filenames