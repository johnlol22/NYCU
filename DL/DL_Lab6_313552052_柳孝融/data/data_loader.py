import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class iCLEVRDataset(Dataset):
    def __init__(self, json_path, image_folder, object_json_path, transform=None, train=True):
        """
        Args:
            json_path: Path to the json file with annotations
            image_folder: Directory with all the images
            object_json_path: Path to the object json file with label-to-idx mapping
            transform: Optional transform to be applied on a sample
            train: Whether this is the training set
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, dict):
            # Train data format: {"filename.png": ["label1", "label2"]}
            self.data = data
            self.filenames = list(self.data.keys())
            self.is_dict = True
        else:
            # Test data format 2: [["label1", "label2"], [...]]
            self.data = data
            self.filenames = [f"{i}.png" for i in range(len(data))]
            self.is_dict = False
            self.is_list_of_dicts = False
        
        with open(object_json_path, 'r') as f:
            self.object_dict = json.load(f)
        
        self.image_folder = image_folder
        self.transform = transform
        self.train = train
        
        # Create label mappings
        self.label_to_idx = {label: idx for idx, label in enumerate(self.object_dict)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.object_dict)}
        self.num_classes = len(self.object_dict)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        
        # Get labels based on data format
        if self.is_dict:
            # Format: {"filename.png": ["label1", "label2"]}
            labels = self.data[file_name]
        else:
            # Format: [["label1", "label2"], [...]]
            labels = self.data[idx]
        
        # Create one-hot encoded label tensor
        label_tensor = torch.zeros(self.num_classes)
        for label in labels:
            if label in self.label_to_idx:  # Handle case where a label might not be in the dictionary
                label_tensor[self.label_to_idx[label]] = 1.0
            else:
                print(f"Warning: Label '{label}' not found in label dictionary.")
        
        if self.train:
            # Load image for training
            img_path = os.path.join(self.image_folder, file_name)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label_tensor
        else:
            # For testing, just return the labels and filename
            return label_tensor, file_name

def get_data_loaders(config):
    """
    Create data loaders for training and testing
    
    Args:
        config: Configuration object with dataset parameters
        
    Returns:
        train_loader, test_loader, new_test_loader, num_classes
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    train_dataset = iCLEVRDataset(
        config.TRAIN_JSON, 
        config.IMAGE_DIR, 
        config.OBJECT_JSON, 
        transform=transform, 
        train=True
    )
    
    test_dataset = iCLEVRDataset(
        config.TEST_JSON, 
        config.IMAGE_DIR, 
        config.OBJECT_JSON, 
        transform=None, 
        train=False
    )
    
    new_test_dataset = iCLEVRDataset(
        config.NEW_TEST_JSON, 
        config.IMAGE_DIR, 
        config.OBJECT_JSON, 
        transform=None, 
        train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    new_test_loader = DataLoader(
        new_test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, test_loader, new_test_loader, train_dataset.num_classes, train_dataset.label_to_idx