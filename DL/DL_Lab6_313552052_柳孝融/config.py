import torch
import os

class Config:
    # Paths
    DATASET_DIR = "../../../iclevr"
    DATA_DIR = "./file"
    TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
    TEST_JSON = os.path.join(DATA_DIR, "test.json")
    NEW_TEST_JSON = os.path.join(DATA_DIR, "new_test.json")
    OBJECT_JSON = os.path.join(DATA_DIR, "objects.json")
    IMAGE_DIR = os.path.join(DATASET_DIR)
    CHECKPOINT_DIR = "checkpoints/UNet_cos_2"
    RESULTS_DIR = "results/UNet_cos_2"
    
    # Model parameters
    IMAGE_SIZE = 64
    CHANNELS = 3
    TIME_EMBEDDING_DIM = 256 
    CONDITION_DIM = 24  # Will be overridden by actual number of classes
    
    # Diffusion parameters
    NOISE_STEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # Evaluation parameters
    GUIDANCE_SCALE = 1.0
    VIS_STEPS = 8
    
    # Specific labels for visualization
    SPECIFIC_LABELS = ["red sphere", "cyan cylinder", "cyan cube"]