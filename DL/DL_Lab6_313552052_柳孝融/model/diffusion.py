import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

class Diffusion:
    """Implementation of the diffusion process"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        """
        Initialize the diffusion process
        
        Args:
            noise_steps: Number of noise steps
            beta_start: Start value for beta schedule
            beta_end: End value for beta schedule
            img_size: Size of the images
            device: Device to run on
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        # Define beta schedule
        self.betas = self.cosine_beta_schedule()
        # self.betas = self.linear_beta_schedule()
        
        # Pre-calculate diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)                        # cumulative product, yi = x1*x2*...*xi
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)   # except last one, creating shift version: 1, x1, x1*x2, ..., x1*...*xi-1
        
        # sqrt(alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)                  # sqrt(a_bar)
        # sqrt(1 - alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)   # sqrt(1-a_bar), used for corruption process
        
        # sqrt(1/alphas) reciprocal
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)     # deterministic variance
    
    def cosine_beta_schedule(self, s=0.008):
        """
        Create a cosine noise schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'

        Args:
            s: Small offset to prevent betas from being too small near t=0

        Returns:
            Tensor of betas for the cosine schedule
        """
        timesteps = self.noise_steps
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, device=self.device) / timesteps
        alphas_cumprod = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to 1.0 at t=0

        # Calculate betas from cumulative product of alphas
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas

        return torch.clip(betas, 0, 0.999)
    
    def linear_beta_schedule(self):
        """Linear noise schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to an image
        
        Args:
            x_start: Starting clean image
            t: Timestep
            noise: Noise to add (or generate if None)
            
        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]                        # 4d sqrt(alpha1*...*alpha_t)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        # Forward process: q(x_t | x_0)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, condition, noise=None):
        """
        Calculate loss for training
        
        Args:
            denoise_model: Model for denoising
            x_start: Starting clean image
            t: Timestep
            condition: Condition tensor
            noise: Noise to add (or generate if None)
            
        Returns:
            MSE loss between predicted and actual noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)           # random number from a normal distribution with mean 0 and variance 1
            
        # Add noise to the input
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict the noise using the model
        noise_pred = denoise_model(x_noisy, t, condition)
        
        # Calculate loss
        return F.mse_loss(noise, noise_pred)
    
    def p_sample(self, model, x, t, condition, t_index):
        """
        Sample from the model at timestep t
        
        Args:
            model: Denoising model
            x: Current noisy image
            t: Current timestep tensor
            condition: Condition tensor
            t_index: Current timestep index (integer)
            
        Returns:
            Denoised image at timestep t-1
        """
        with torch.no_grad():
            # Reshape for broadcasting
            betas_t = self.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]    # sqrt(1-alpha_bar)
            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]    #  sqrt(1/alpha)
            
            # Predict the noise
            predicted_noise = model(x, t, condition)
            
            # Calculate the mean of p(x_{t-1} | x_t)
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
            )
            
            # Add noise if not the final step
            if t_index == 0:
                return model_mean
            else:
                posterior_variance_t = self.posterior_variance[t][:, None, None, None]
                noise = torch.randn_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample(self, model, condition, batch_size=16, channels=3):
        """
        Generate samples from scratch using the model (the process of acquiring knowledge or skills without any prior experience)
        
        Args:
            model: Denoising model
            condition: Condition tensor
            batch_size: Number of images to generate
            channels: Number of channels in the image
            
        Returns:
            Generated images
        """
        model.eval()
        
        # Start from pure noise
        shape = (condition.shape[0], channels, self.img_size, self.img_size)
        img = torch.randn(shape, device=self.device)        # pure gaussian noise
        
        # Progressively denoise
        for i in tqdm(reversed(range(0, self.noise_steps)), desc='Sampling'):
            # Create timestep tensor
            t = torch.full((condition.shape[0],), i, device=self.device, dtype=torch.long)      # condition.shape[0] refers to batch size, each needs 
                                                                                                # its own timestep value, but all images are at the same denoising loop i.
            
            # Sample from p(x_{t-1} | x_t)
            img = self.p_sample(model, img, t, condition, i)
        
        # Return images in [-1, 1] range
        return img