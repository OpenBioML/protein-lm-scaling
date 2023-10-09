import pytest
import torch
from torch.nn import functional as F

from model_pytorch import APTAttention

class ParameterConfig:
    def __init__(self):
        self.max_position_embeddings = 512
        self.position_embedding = 'rope'
        self.max_sequence_length = 512
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.scale_attn_weights = True
        self.scale_attn_by_inverse_layer_idx = True
        self.reorder_and_upcast_attn = True
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1


def test_gqa_attn():
    # 1. Initialize with mock config
    config = ParameterConfig()
    attention = APTAttention(config, is_cross_attention=False, layer_idx=0)
    
    # 2. Generate random input tensors
    batch_size = 4
    seq_length = 100
    num_heads = config.num_attention_heads  # Using the number of attention heads from config
    query_dim = config.hidden_size // num_heads
    query = torch.randn(batch_size, num_heads, seq_length, query_dim)
    key = torch.randn(batch_size, num_heads, seq_length, query_dim)
    value = torch.randn(batch_size, num_heads, seq_length, query_dim)
    
    # Create a random attention mask (if required)
    attention_mask = torch.ones(batch_size, 1, seq_length, seq_length)
    padding_positions = 10
    attention_mask[:, :, -padding_positions:, :] = float('-inf')
    attention_mask[:, :, :, -padding_positions:] = float('-inf')
    
    # 3. Pass them through the _gqa_attn method
    attn_output, attn_weights = attention._gqa_attn(query, key, value, attention_mask=attention_mask)
    
    # 4. Check the shapes and types of the output
    assert isinstance(attn_output, torch.Tensor)
    assert attn_output.shape == (batch_size, num_heads, seq_length, query_dim)
    assert isinstance(attn_weights, torch.Tensor)
    assert attn_weights.shape == (batch_size, num_heads, seq_length, seq_length)
    print("Test passed!")

test_gqa_attn()


