import torch.nn as nn
import torch
import math
import torch.nn.functional as F
#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads        # each head processes the entire image, but focuses on different aspects or patterns
        self.dim = dim
        self.head_dim = dim // num_heads  # 768 // 16 = 48, 48 dimensions per head, each head operates on a subset of the full dimension
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(dim, dim)           # create a learnable weight matrix of shape (dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout for attention weights
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Check if dimensions are compatible
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
    
    def forward(self, x):
        ''' 
        Input x tensor shape is (batch_size, num_image_tokens, dim)
        because the bidirectional transformer first will embed each token to dim dimension, 
        and then pass to n_layers of encoders consist of Multi-Head Attention and MLP.
        # of head set 16
        Total d_k, d_v set to 768
        d_k, d_v for one head will be 768//16=48.
        '''
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape for multi-head attention
        # Shape after projection: (batch_size, seq_len, dim)
        q = self.q_proj(x)      # multiply the weight matrix, seq_len refers to the number of image tokens
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)      # reshape, dim into num_head and head_dim, that is, 768 -> 16x48
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # transpose(-2, -1) transposes last two dimensions
        # in this way, q last two size (seq_len, head_dim) will match k (head_dim, seq_len)
        # sqrt part is the scaling part, in order to prevent exploding gradient and stabilize softmax function
        # as the dim of q and k increase, dot product grow larger, without scaling, the large values will push the softmax into regions with extremely small gradient
        # causing the vanishing gradient problem 
        
        # Apply softmax to get attention weights
        # dim means doing softmax to specific dimension, 0 means the same pos in each dimensions, 1 means col, 2 means row, -1 is the same as 2
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout to attention weights, prevent overfitting by randomly dropping attention connections
        # improved generalization, create an implict ensemble of different attention patterns
        # prevent co-adaptation, encourages the heads to learn diverse, complementary features
        # robustness, makes the model less sensitive to individual attention values
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(attn_weights, v)     # v shape (batch_size, num_heads, seq_len, head_dim), attn_weights shape (batch_size, num_heads, seq_len, seq_len)
        
        # Transpose back and reshape to original dimensions
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)         # contigous return a contiguous in memory tensor containing the same data as self tensor
        
        # Final linear projection, multiply with W^O
        out = self.out_proj(out)
        
        return out

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    