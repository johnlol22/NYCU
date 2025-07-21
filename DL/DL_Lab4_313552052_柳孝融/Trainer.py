import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10




def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size, epsilon=1e-8):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().clamp(min=epsilon))
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.anneal_type = args.kl_anneal_type
        self.n_cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.n_iter = args.num_epoch
        self.current_epoch = current_epoch
        
        # Initialize beta based on annealing type
        if self.anneal_type == 'Cyclical':
            self.beta_values = self.frange_cycle_linear(self.n_iter, start=0.0, stop=1.0, 
                                                  n_cycle=self.n_cycle, ratio=self.ratio)
            self.beta = self.beta_values[self.current_epoch]
        elif self.anneal_type == 'Monotonic':
            self.beta = min(1.0, self.current_epoch / (self.n_iter // 2))
        else:  # No annealing
            self.beta = 1.0
        
    def update(self):
        self.current_epoch += 1
        if self.current_epoch >= self.n_iter:
            # Cap at final epoch
            self.current_epoch = self.n_iter - 1

        if self.anneal_type == 'Cyclical':
            self.beta = self.beta_values[self.current_epoch]
        elif self.anneal_type == 'Monotonic':
            self.beta = min(1.0, self.current_epoch / (self.n_iter // 2))
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        """
        Creates a cyclical schedule for KL annealing
        """
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # step is based on ratio
        
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        # self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.optim = optim.AdamW(
            self.parameters(), 
            lr=self.args.lr,
            weight_decay=1e-4,  # You can tune this parameter (typically between 1e-5 and 1e-2)
            betas=(0.9, 0.999)  # Default values, but you might experiment with these
        )
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[20, 50, 80, 110], gamma=0.5)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        B, F, C, H, W = img.shape

        # Initial frame
        x_prev = img[:, 0]
        generated_frames = [x_prev]

        for i in range(1, F):
            p_curr = label[:, i]

            # Encode previous frame and current label
            f_prev = self.frame_transformation(x_prev)
            p_feat = self.label_transformation(p_curr)

            # Sample from prior for inference
            z_shape = f_prev.shape
            z = torch.randn(B, self.args.N_dim, z_shape[2], z_shape[3], device=self.args.device)

            # Decoder fusion
            decoder_output = self.Decoder_Fusion(f_prev, p_feat, z)

            # Generate current frame
            x_hat = self.Generator(decoder_output)
            generated_frames.append(x_hat)

            # Use generated frame for next iteration
            x_prev = x_hat

        # Stack generated frames along frame dimension
        generated_sequence = torch.stack(generated_frames, dim=1)

        return generated_sequence
    
    def training_stage(self):
        train_losses = []
        val_losses = []
        for i in range(self.args.num_epoch):
            train_loss = 0
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            try :
                for (img, label) in (pbar := tqdm(train_loader, ncols=150)):
                    img = img.to(self.args.device)
                    label = label.to(self.args.device)
                    loss = self.training_one_step(img, label, adapt_TeacherForcing)
                    train_loss += loss.item()
                    beta = self.kl_annealing.get_beta()
                    if adapt_TeacherForcing:
                        self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                    else:
                        self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                
                if self.current_epoch % self.args.per_save == 0:
                    self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
                val_loss = self.eval()
                self.current_epoch += 1
                self.scheduler.step()
                self.teacher_forcing_ratio_update()
                self.kl_annealing.update()
                print(train_loss/len(train_loader), file=train_loss_file)
                print(val_loss, file=val_loss_file)
                # train_losses.append(train_loss/len(train_loader))
                # val_losses.append(val_loss)
            except Exception as e:
                continue
        # self.plot(train_losses, val_losses)
            
            
    @torch.no_grad()
    def eval(self):
        val_loss = 0
        num_batches = 0
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            val_loss+=loss.item()
            num_batches+=1
        return val_loss / num_batches if num_batches > 0 else 0
    
    # Implementation for training_one_step
    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.optim.zero_grad()

        B, F, C, H, W = img.shape  # Batch, Frames, Channels, Height, Width

        # Prepare storage for losses
        mse_losses = 0
        kl_losses = 0

        # Initial frame is always provided
        x_prev = img[:, 0]  # First frame

        # Iterate through the frames sequence
        for i in range(1, F):
            # Get current frame and pose label
            x_curr = img[:, i]
            p_curr = label[:, i]

            # Encode frames and labels
            f_prev = self.frame_transformation(x_prev)
            p_feat = self.label_transformation(p_curr)

            # For posterior prediction
            f_curr = self.frame_transformation(x_curr)

            # Concatenate current features and pose
            # post_input = torch.cat([f_curr, p_feat], dim=1)

            # Predict mean and log variance for the latent distribution
            z, mu, logvar = self.Gaussian_Predictor(f_curr, p_feat)

            # Sample noise using reparameterization trick
            # z = self.Gaussian_Predictor.reparameterize(mu, logvar)

            # Decoder fusion takes previous frame feature, pose feature, and noise
            # decoder_input = torch.cat([f_prev, p_feat, z], dim=1)
            decoder_output = self.Decoder_Fusion(f_prev, p_feat, z)

            # Generate current frame
            x_hat = self.Generator(decoder_output)

            # Calculate MSE loss for reconstruction
            # In training_one_step
            mse_loss = self.mse_criterion(x_hat, x_curr)
            # l1_loss = nn.functional.l1_loss(x_hat, x_curr)
            # mse_losses += 0.8 * mse_loss + 0.2 * l1_loss
            mse_losses += mse_loss
            # Calculate KL divergence loss
            kl_loss = kl_criterion(mu, logvar, B)
            kl_losses += kl_loss

            # frame_tfr = self.tfr * (1 - (i / F) * 0.5)  # Decay by up to 50% for later frames
            # adapt_TeacherForcing = random.random() < frame_tfr
            # For the next iteration, use either ground truth (teacher forcing) or prediction
            if adapt_TeacherForcing:
                x_prev = x_curr  # Use ground truth
            else:
                x_prev = x_hat.detach()  # Use model's prediction

        # Apply KL annealing weight to KL loss
        beta = self.kl_annealing.get_beta()
        loss = mse_losses + beta * kl_losses
        if torch.isnan(loss):
            print(f"NaN detected at epoch {self.current_epoch}, beta: {beta}")
            print(f"mse_losses: {mse_losses.item()}, kl_losses: {kl_losses.item()}")
            print(f"mu_range: {mu.min().item()}-{mu.max().item()}, logvar_range: {logvar.min().item()}-{logvar.max().item()}")
            # You might want to skip the backward pass to avoid propagating NaNs
            return loss

        # Backpropagation
        loss.backward()
        self.optimizer_step()

        return loss
    
    # Implementation for val_one_step
    def val_one_step(self, img, label):
        # Similar to training but without teacher forcing or optimization
        B, F, C, H, W = img.shape
        mse_losses = 0
        kl_losses = 0

        # Initial frame
        x_prev = img[:, 0]

        # Store generated frames for visualization if needed
        generated_frames = [x_prev]

        # PSNR values for each frame
        psnr_values = []

        for i in range(1, F):
            p_curr = label[:, i]
            x_curr = img[:, i]  # Ground truth for current frame

            # Encode previous frame and current label
            f_prev = self.frame_transformation(x_prev)
            p_feat = self.label_transformation(p_curr)

            # For inference, we don't have access to current frame
            # So we sample from the prior distribution N(0, I)
            z_shape = f_prev.shape
            z = torch.randn(B, self.args.N_dim, z_shape[2], z_shape[3], device=self.args.device)

            # Decoder fusion
            # decoder_input = torch.cat([f_prev, p_feat, z], dim=1)
            decoder_output = self.Decoder_Fusion(f_prev, p_feat, z)

            # Generate current frame
            x_hat = self.Generator(decoder_output)

            # Calculate MSE and PSNR for evaluation
            mse_loss = self.mse_criterion(x_hat, x_curr)
            mse_losses += mse_loss

            # Calculate PSNR
            psnr = Generate_PSNR(x_hat, x_curr)
            psnr_values.append(psnr.item())

            # For next iteration, always use model's prediction in validation
            x_prev = x_hat

            # Store generated frame
            generated_frames.append(x_hat)

        # Create GIF if storing visualization is enabled
        if self.args.store_visualization and self.current_epoch % self.args.per_save == 0:
            # Convert tensor to image format for GIF
            vis_frames = [frame[0].cpu() for frame in generated_frames]  # Take first batch item
            self.make_gif(vis_frames, os.path.join(self.args.save_root, f"val_epoch_{self.current_epoch}.gif"))

            # Optionally save PSNR plot
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(psnr_values) + 1), psnr_values)
            plt.title(f'PSNR per frame - Epoch {self.current_epoch}')
            plt.xlabel('Frame')
            plt.ylabel('PSNR (dB)')
            plt.savefig(os.path.join(self.args.save_root, f"psnr_epoch_{self.current_epoch}.png"))
            plt.close()

        return mse_losses
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # Decay teacher forcing ratio after specified epoch
        if self.current_epoch >= self.args.tfr_sde:
            self.tfr = max(0, self.tfr - self.args.tfr_d_step)
            
    # def teacher_forcing_ratio_update(self):
    #     # Base decay after specified epoch
    #     if self.current_epoch >= self.args.tfr_sde:
    #         # Standard decay
    #         standard_tfr = max(0, self.tfr - self.args.tfr_d_step)
    # 
    #         # Option 1: Simple decay
    #         # self.tfr = standard_tfr
    # 
    #         # Option 2: Cyclical teacher forcing
    #         # cycle_length = 10
    #         # cycle_position = (self.current_epoch - self.args.tfr_sde) % cycle_length
    #         # if cycle_position == 0 and self.current_epoch > self.args.tfr_sde + cycle_length:
    #         #     # Reset to higher value at start of cycle, but not as high as initial value
    #         #     self.tfr = min(0.7, standard_tfr + 0.3)
    #         # else:
    #         #     self.tfr = standard_tfr
    # 
    #         # Option 3: Validation-based adaptation
    #         if self.current_epoch > 0 and len(self.val_losses) >= 2:
    #             # If validation loss is increasing, slow down decay
    #             if self.val_losses[-1] > self.val_losses[-2]:
    #                 self.tfr = min(0.9, standard_tfr + 0.1)
    #             else:
    #                 self.tfr = standard_tfr
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optim.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            # self.args.lr = checkpoint['lr']
            self.args.lr = self.args.lr
            self.tfr = checkpoint['tfr']
            
            # self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.optim      = optim.AdamW(
                self.parameters(), 
                lr=self.args.lr,
                weight_decay=1e-2,  # You can tune this parameter (typically between 1e-5 and 1e-2)
                betas=(0.9, 0.999)  # Default values, but you might experiment with these
            )
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10,25], gamma=0.25) # 20, 50, 80
            self.kl_annealing = kl_annealing(self.args, current_epoch=0)
            self.current_epoch = checkpoint['last_epoch']
            # self.current_epoch = 0

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

    def plot(self, train_losses, val_losses):
        # Plot training and validation losses
        train_losses = [loss if isinstance(loss, (int, float)) else loss.cpu().item() for loss in train_losses]
        val_losses = [loss if isinstance(loss, (int, float)) else loss.cpu().item() for loss in val_losses]


        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('loss_plot_cyc7.png')

        return

def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        val_loss = model.eval()
        print(f'testing result, loss: {val_loss}')
    else:
        train_loss_path = './result/cyc7/train_loss.txt'
        val_loss_path = './result/cyc7/val_loss.txt'
        train_loss_file = open(train_loss_path, 'w')
        val_loss_file = open(val_loss_path, 'w')
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=1e-4,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=1)
    parser.add_argument('--num_epoch',     type=int, default=150,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=70,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=5,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.3,              help="")
    # controls the proportion of each of the KL annealing cycle devoted to increasing the KL weight, for example ratio = 0.5
    # half of the cycle increasing beta from 0 to 1, then reamins at 1 for the second half
    # lower means train more with maximum KL regularization, more stable, less exploration, 
    # higher means gentle increase in regularization, more initial freedom for learning reconstruction
    

    

    args = parser.parse_args()
    
    main(args)
