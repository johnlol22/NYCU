from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q2_config = BaseQuantizeConfig(nbits=2, group_size=128)  # Aggressive quantization (2-bit)
    q3_config = BaseQuantizeConfig(nbits=3, group_size=128)  # Moderate quantization (3-bit)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=128)  # Moderate quantization (4-bit)
    q6_config = BaseQuantizeConfig(nbits=6, group_size=128)  # Moderate quantization (4-bit)
    q8_config = BaseQuantizeConfig(nbits=8, group_size=256)  # Light quantization (8-bit)

    # Quantize the patch embedding and position embedding with less aggressive settings
    quant_config['patch_embed.proj'] = q4_config
    quant_config['pos_embed'] = q4_config
    
    # Quantize the transformer blocks
    for i in range(n_blocks):
        # Attention layers - using different configs based on sensitivity
        quant_config[f'blocks.{i}.attn.qkv'] = q4_config       # Query, Key, Value projections
        quant_config[f'blocks.{i}.attn.proj'] = q8_config      # Output projection
        
        # MLP layers - typically more resilient to quantization
        quant_config[f'blocks.{i}.mlp.fc1'] = q8_config        # First fully connected layer
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_config        # Second fully connected layer
        
        # Layer norms - using higher precision as they're sensitive
        quant_config[f'blocks.{i}.norm1'] = q8_config
        quant_config[f'blocks.{i}.norm2'] = q8_config
    
    # Head layer (classification) - typically needs higher precision
    quant_config['norm'] = q8_config
    quant_config['head'] = q8_config
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    
    # Different quantization configurations
    q2_config = BaseQuantizeConfig(nbits=2, group_size=256)  # Most aggressive
    q4_config = BaseQuantizeConfig(nbits=4, group_size=256)  # More precise
    q5_config = BaseQuantizeConfig(nbits=5, group_size=256)  # More precise
    q6_config = BaseQuantizeConfig(nbits=6, group_size=256)  # More precise
    q8_config = BaseQuantizeConfig(nbits=8, group_size=256)  # More precise
    
    
    # n_layers = model.config.num_hidden_layers
    # q2_config = BaseQuantizeConfig(nbits=2, group_size=64) 
    
    # Embeddings need higher precision
    quant_config['model.embed_tokens'] = q4_config
    
    # Layer norm parameters are small but critical - keep at higher precision
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.input_layernorm'] = q4_config
        quant_config[f'model.layers.{i}.post_attention_layernorm'] = q4_config
    
    # First and last layers are more sensitive to quantization
    # First layer
    quant_config['model.layers.0.self_attn.q_proj'] = q4_config
    quant_config['model.layers.0.self_attn.k_proj'] = q4_config
    quant_config['model.layers.0.self_attn.v_proj'] = q4_config
    quant_config['model.layers.0.self_attn.o_proj'] = q4_config
    quant_config['model.layers.0.mlp.gate_proj'] = q6_config
    quant_config['model.layers.0.mlp.up_proj'] = q6_config
    quant_config['model.layers.0.mlp.down_proj'] = q8_config
    
    # Last layer
    quant_config[f'model.layers.{n_layers-1}.self_attn.q_proj'] = q4_config
    quant_config[f'model.layers.{n_layers-1}.self_attn.k_proj'] = q4_config
    quant_config[f'model.layers.{n_layers-1}.self_attn.v_proj'] = q4_config
    quant_config[f'model.layers.{n_layers-1}.self_attn.o_proj'] = q4_config
    quant_config[f'model.layers.{n_layers-1}.mlp.gate_proj'] = q6_config
    quant_config[f'model.layers.{n_layers-1}.mlp.up_proj'] = q6_config
    quant_config[f'model.layers.{n_layers-1}.mlp.down_proj'] = q8_config
    
    # Middle layers can use more aggressive quantization
    for i in range(1, n_layers-1):
        # Attention is usually more sensitive than MLP
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q6_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q6_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q6_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q6_config
        
        # MLP can use more aggressive quantization
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_config
    
    # Final norm and output embedding need higher precision
    quant_config['model.norm'] = q4_config
    quant_config['lm_head'] = q4_config
    
    return quant_config