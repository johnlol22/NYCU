import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.args = args
        self.device = args.device
        self.best_loss = float('inf')
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
        # Load checkpoint if specified
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            if self.scheduler and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Loaded checkpoint. Best loss: {self.best_loss}")
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints/square", exist_ok=True)

    def train_one_epoch(self, epoch, train_loader):
        self.model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for i, imgs in enumerate(train_bar):
            imgs = imgs.to(self.device)
            
            # Forward pass
            # logits size [16, 256, 1025] target size [16, 256]
            # 16 is batch size, 256 is sequence length, 1025 is 1024 tokens plus a mask token
            logits, target = self.model(imgs)


            # Calculate loss
            # -1 means the size is depending on another param
            # so logits reshape to [16x256, 1025], target reshape to [16x256]
            # model has to learn the probabilities of 1025
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            # Backward pass with gradient accumulation
            # train with effectively larger batch size, more stable gradient estimation
            # for example, accum_grad 10 means effectively training with a batch size 10 times larger than loading into memory
            loss = loss / self.args.accum_grad  # Normalize by accumulation steps
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % self.args.accum_grad == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Add this line
                self.optim.step()
                self.optim.zero_grad()
            
            # Update running loss
            # multiply back for truly loss
            running_loss += loss.item() * self.args.accum_grad
            train_bar.set_postfix(loss=running_loss/(i+1))
            
        # calculate the loss per batch, making it independent of dataset size
        epoch_loss = running_loss / len(train_loader)
            
        return epoch_loss

    def eval_one_epoch(self, epoch, val_loader):
        self.model.eval()
        running_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for i, imgs in enumerate(val_bar):
                imgs = imgs.to(self.device)
                
                # Forward pass
                logits, target = self.model(imgs)
                
                # Calculate loss
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                
                # Update running loss
                running_loss += loss.item()
                val_bar.set_postfix(loss=running_loss/(i+1))
        
        # don't consider gradient accumulation cuz no gradient computation
        epoch_loss = running_loss / len(val_loader)
        return epoch_loss

    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.transformer.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),      # controls the exponential moving average rates for the first and second moment estimates (momentum and variance) of the gradient
            weight_decay=5e-3
        )

        scheduler = None
        # # Learning rate scheduler with cosine annealing
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.args.epochs,                 # the length of complete cycle to reach eta_min
        #     eta_min=self.args.learning_rate * 0.1   # gradually decreasing to this value
        # )
        return optimizer, scheduler
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        # Save just the transformer weights directly
        checkpoint_path = os.path.join("transformer_checkpoints/square", f"epoch_{200+epoch}.pt")
        
        # Save only the transformer state dict
        torch.save(self.model.transformer.state_dict(), checkpoint_path)
        print(f"Transformer weights saved to {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_path = os.path.join("transformer_checkpoints/square", "best_model_weight.pt")
            torch.save(self.model.transformer.state_dict(), best_path)
            print(f"Best transformer weights saved to {best_path}")
        
        # save the full training state in a different file
        # if resuming training later
        full_checkpoint_path = os.path.join("transformer_checkpoints/square", f"full_epoch_{200+epoch}.pt")
        full_checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss
        }
        
        if self.scheduler:
            full_checkpoint['scheduler'] = self.scheduler.state_dict()
        
        torch.save(full_checkpoint, full_checkpoint_path)
    
    def generate_samples(self, epoch, num_samples=4):
        """Generate samples to visualize training progress"""
        self.model.eval()
        with torch.no_grad():
            # Generate tokens
            z_indices, _ = self.model.inpainting()  # z_indices [1, 256]
            
            # Decode tokens to images using VQGAN decoder
            shape = (1, 16, 16, 256)  # Adjust shape for your model
            z_q = self.model.vqgan.codebook.embedding(z_indices).view(shape)
            z_q = z_q.permute(0, 3, 1, 2)
            torch.cuda.empty_cache()
            reconstructions = self.model.vqgan.decoder(z_q)
            
            # Save generated images
            samples_dir = os.path.join("transformer_checkpoints/square", "samples")
            os.makedirs(samples_dir, exist_ok=True)
            vutils.save_image(
                reconstructions.data,
                os.path.join(samples_dir, f"epoch_{200+epoch}.png"),
                normalize=True,
                nrow=int(num_samples**0.5)
            )


def plot(train_losses, val_losses):
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot6_1_4.png')
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='partial: Only train pat of the dataset')    
    parser.add_argument('--accum-grad', type=int, default=16, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='./config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,     # drop the last incomplete batch
                                pin_memory=True,    # data loader will copy tensors into device pinned memory before returning them, for efficiency data transfer
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)

    
     # Define a warmup schedule
    def lr_lambda(current_step: int):
        # Warmup for first 10% of training
        warmup_steps = int(0.1 * args.epochs * len(train_loader))
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay for the rest
        decay_steps = args.epochs * len(train_loader) - warmup_steps
        decay_factor = 0.1  # Final lr ratio
        step = current_step - warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        return max(decay_factor, cosine_decay)


    train_transformer.scheduler = torch.optim.lr_scheduler.LambdaLR(train_transformer.optim, lr_lambda)

    train_losses=[]
    val_losses=[]
    #TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        # Train for one epoch
        train_loss = train_transformer.train_one_epoch(epoch, train_loader)
        if train_transformer.scheduler:
            train_transformer.scheduler.step()
        print(f"Train Loss: {train_loss:.4f}")
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = train_transformer.eval_one_epoch(epoch, val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        
        # Check if this is the best model
        is_best = val_loss < train_transformer.best_loss
        if is_best:
            train_transformer.best_loss = val_loss
            print(f"New best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_per_epoch == 0 or is_best or epoch == args.epochs:
            train_transformer.save_checkpoint(epoch, val_loss, is_best)
        
        # Generate samples every 10 epochs to visualize progress
        if epoch % 10 == 0 or epoch == args.epochs:
            train_transformer.generate_samples(epoch)
            
    print("Training complete!")
    plot(train_losses, val_losses)