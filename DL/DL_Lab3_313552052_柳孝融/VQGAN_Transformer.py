import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        """
        Encodes the input image x to latent space using VQGAN encoder
        
        Args:
            x: Input image tensor
        
        Returns:
            z_indices: Discrete token indices (quantized latent codes)
        """
        # Get the latent representation, which means the result of encoder, before quantization
        latent = self.vqgan.encoder(x)
        quantized_latent = self.vqgan.quant_conv(latent)
        
        # Quantize the latent representation to get indices
        # Adapting to the VQGAN implementation structure
        z_q, z_indices, _ = self.vqgan.codebook(quantized_latent)
        
        '''
        # Instead of using a fixed reshape, check the actual shape of z_indices
        # and flatten the spatial dimensions
        if len(z_indices.shape) == 4:  # [B, C, H, W]
            z_indices = z_indices.reshape(x.shape[0], -1)
        elif len(z_indices.shape) == 3:  # [B, H, W]
            z_indices = z_indices.reshape(x.shape[0], -1)
        elif len(z_indices.shape) == 1:  # Already flattened but size 1
            # Create a properly sized tensor for the batch
            z_indices = z_indices.unsqueeze(0).expand(x.shape[0], -1)
        '''
        return latent, z_q
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.        percentage of tokens that should remain masked, should not proportional to ratio

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            def func(ratio):
                return 1.0 - ratio
            return func
        elif mode == "cosine":
            def func(ratio):
                return np.cos(ratio * np.pi / 2)
            return func
        elif mode == "square":
            def func(ratio):
                return 1.0 - ratio ** 2
            return func
        elif mode == "logarithmic":
            def func(ratio):
                # Log schedule ensures more tokens are kept unmasked during early iterations
                # and then rapidly increases masking as ratio approaches 1
                # Adding epsilon to avoid log(0)
                epsilon = 1e-6
                return -np.log(ratio + epsilon) / (-np.log(epsilon))
            return func
        else:
            raise NotImplementedError


##TODO2 step1-3:            
    def forward(self, x):
        """
        Forward pass for training using completely random masking
    
        Args:
            x: Input image tensor
        
        Returns:
            logits: Transformer's token predictions
            z_indices: Ground truth token indices
        """
        batch_size = x.shape[0]  # Use input batch size as reference
    
        # Get token indices from VQGAN
        latent, z_q = self.encode_to_z(x)
        _, z_indices, _ = self.vqgan.codebook(self.vqgan.quant_conv(latent))
        z_indices = z_indices.reshape(batch_size, -1)   # size [batch_size, 256]
    
        # Use a truly random masking ratio between 10% and 90%
        # This wide range ensures the model learns to handle various masking conditions
        rand_ratio = 0.1 + torch.rand(1).item() * 0.8  # Random between 0.1 and 0.9
    
        # Determine number of tokens to mask
        num_tokens_masked = int(self.num_image_tokens * rand_ratio)
    
        # For each sample in the batch, create a unique random mask
        masked_indices = []
        for i in range(batch_size):
            # Randomly select which tokens to mask
            mask = torch.zeros(self.num_image_tokens, dtype=torch.bool, device=x.device)
            mask_positions = torch.randperm(self.num_image_tokens, device=x.device)[:num_tokens_masked]
            mask[mask_positions] = True
            masked_indices.append(mask)
        
        # Stack masks for the batch
        masked_indices = torch.stack(masked_indices)
    
        # Create masked input (replace masked tokens with mask_token_id)
        transformer_input = z_indices.clone()
    
        # Apply masking - make sure the dimensions match
        for i in range(batch_size):
            transformer_input[i][masked_indices[i]] = self.mask_token_id
    
        # Feed masked input to transformer to predict all tokens
        logits = self.transformer(transformer_input)
    
        # Return logits and ground truth indices
        return logits, z_indices
    
    ##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, current_z_indices=None, current_mask=None, ratio=0.5):
        """
        Performs one step of iterative decoding for image generation
        
        Args:
            current_z_indices: Current token indices (if None, will initialize new ones)
            current_mask: Current mask (if None, will initialize all tokens as masked)
            ratio: Current progress ratio (0 to 1)
            
        Returns:
            z_indices_predict: Updated token indices after this step
            mask_bc: Final mask indicating which tokens were generated
        """
        device = next(self.transformer.parameters()).device
        batch_size = 1  # For inference, typically batch_size=1
        
        # Use provided tokens and mask if available, otherwise initialize
        if current_z_indices is None:
            z_indices = torch.ones(batch_size, self.num_image_tokens, 
                                  dtype=torch.long, device=device) * self.mask_token_id
        else:
            z_indices = current_z_indices
            
        if current_mask is None:
            mask = torch.ones(batch_size, self.num_image_tokens, 
                             dtype=torch.bool, device=device)
        else:
            mask = current_mask
        
        # Get predictions from transformer
        logits = self.transformer(z_indices)
        
        # Apply softmax to convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get the highest probability and its corresponding token for each position
        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)
        
        # Apply temperature annealing with Gumbel noise as confidence measure
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))
        temperature = self.choice_temperature * (1-ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        # For unmasked tokens, set confidence to infinity to keep them
        confidence[~mask] = float('inf')
        
        # IMPORTANT: In this implementation, gamma returns a value that's
        # used directly to determine how many tokens to unmask.
        # Despite the naming (mask_ratio), it's effectively being used
        # as an unmask_ratio - which works well in practice.
        tokens_to_unmask_ratio = self.gamma(ratio)
        num_tokens_to_unmask = max(1, int(tokens_to_unmask_ratio * self.num_image_tokens))
        
        # Sort confidence to determine which tokens to unmask
        confidence_flat = confidence.reshape(-1)
        _, indices = torch.sort(confidence_flat, descending=True)
        
        # Create new mask for next iteration
        old_mask = mask.clone()
        mask_bc = mask.clone().reshape(-1)
        mask_bc[indices[:num_tokens_to_unmask]] = False
        mask = mask_bc.reshape(batch_size, self.num_image_tokens)
        
        # Update token values with predictions for newly revealed tokens
        new_revealed = (~mask) & old_mask
        z_indices_c = z_indices.clone()
        z_indices_c[new_revealed] = z_indices_predict[new_revealed]
        
        # Before returning, ensure indices are in valid range
        z_indices_c = torch.clamp(z_indices_c, 0, self.mask_token_id-1)
        
        return z_indices_c, mask_bc.reshape(batch_size, self.num_image_tokens)
        '''
        logits = self.transformer(None)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = None

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = None

        ratio=None 
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = None  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        
        mask_bc=None
        return z_indices_predict, mask_bc
        '''
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
