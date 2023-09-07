import os
import math
import functools
import pytest
import torch
from protein_lm.modeling.utils.rotary_embedding import RotaryEmbedding
from protein_lm.modeling.utils.rerope_embedding import RectifiedRotaryEmbedding
from protein_lm.modeling.utils.alibi_embedding import create_alibi_tensor
from protein_lm.modeling.utils.scaled_rope_embedding import LlamaLinearScalingRotaryEmbedding,LlamaDynamicNTKScalingRotaryEmbedding

assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
encodings = ['rope','rerope','alibi','linear_rope_scaling','dynamic_rope_scaling']
@pytest.mark.parametrize("encoding",encodings)
def test_encoding(encoding):
    if encoding == 'rope':
        head_dim = 64 
        seq_len = 10
        heads = 12
        rot_emb = RotaryEmbedding(dim = head_dim)
        q = torch.zeros(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.zeros(1, heads, seq_len,head_dim) # keys

        qr,kr = rot_emb(q,k)
        assert_equal(q,qr)
        assert_equal(k,kr)
        q = torch.ones(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.ones(1, heads, seq_len,head_dim) # keys
        qr1,kr1 = rot_emb(q,k)
        rope_path = os.path.join(os.path.dirname(__file__),'tensors','rope.pkl')
        rope = torch.load(rope_path)
        qr2,kr2 = rope[0],rope[1]
        torch.testing.assert_close(qr1,qr2)
        torch.testing.assert_close(kr1,kr2)
        

    elif encoding == 'rerope':
        head_dim = 64
        seq_len = 10
        rot_emb = RectifiedRotaryEmbedding(dim = head_dim,max_position_embeddings=seq_len )
        heads = 12
        position_ids = torch.arange(0,seq_len,dtype=torch.int32).unsqueeze(0)
        q = torch.zeros(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.zeros(1, heads, seq_len,head_dim) # keys
        qr,kr = rot_emb(q,k,seq_len=seq_len,position_ids = position_ids)
        assert_equal(q,qr)
        assert_equal(k,kr)
        q = torch.ones(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.ones(1, heads, seq_len,head_dim) # keys
        qr1,kr1= rot_emb(q,k,seq_len=seq_len,position_ids = position_ids)
        rerope_path = os.path.join(os.path.dirname(__file__),'tensors','rerope.pkl')
        rerope = torch.load(rerope_path)
        qr2,kr2 = rerope[0],rerope[1]
        torch.testing.assert_close(qr1,qr2)
        torch.testing.assert_close(kr1,kr2)

    elif encoding == 'linear_rope_scaling':
        head_dim = 64
        seq_len = 10
        scaling_factor=1.0
        rope_theta=10000
        rot_emb = LlamaLinearScalingRotaryEmbedding(dim=head_dim,max_position_embeddings=seq_len,scaling_factor=scaling_factor,base=rope_theta)
        heads = 12
        position_ids = torch.arange(0,seq_len,dtype=torch.int32).unsqueeze(0)
        q = torch.zeros(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.zeros(1, heads, seq_len,head_dim) # keys
        qr,kr = rot_emb(q,k,seq_len=seq_len,position_ids = position_ids)
        assert_equal(q,qr)
        assert_equal(k,kr)
        q = torch.ones(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.ones(1, heads, seq_len,head_dim) # keys
        qr1,kr1= rot_emb(q,k,seq_len=seq_len,position_ids = position_ids)
        rerope_path = os.path.join(os.path.dirname(__file__),'tensors','linear_rope.pkl')
        rerope = torch.load(rerope_path)
        qr2,kr2 = rerope[0],rerope[1]
        torch.testing.assert_close(qr1,qr2)
        torch.testing.assert_close(kr1,kr2)

    elif encoding == "dynamic_rope_scaling":
        head_dim = 64
        seq_len = 10
        scaling_factor=1.0
        rope_theta=10000
        rot_emb = LlamaDynamicNTKScalingRotaryEmbedding(dim=head_dim,max_position_embeddings=seq_len,scaling_factor=scaling_factor,base=rope_theta)
        heads = 12
        position_ids = torch.arange(0,seq_len,dtype=torch.int32).unsqueeze(0)
        q = torch.zeros(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.zeros(1, heads, seq_len,head_dim) # keys
        qr,kr = rot_emb(q,k,seq_len=seq_len,position_ids = position_ids)
        assert_equal(q,qr)
        assert_equal(k,kr)
        q = torch.ones(1, heads, seq_len, head_dim) # queries - (batch, heads, seq len, dimension of head)
        k = torch.ones(1, heads, seq_len,head_dim) # keys
        qr1,kr1= rot_emb(q,k,seq_len=seq_len,position_ids = position_ids)
        rerope_path = os.path.join(os.path.dirname(__file__),'tensors','dynamic_rope.pkl')
        rerope = torch.load(rerope_path)
        qr2,kr2 = rerope[0],rerope[1]
        torch.testing.assert_close(qr1,qr2)
        torch.testing.assert_close(kr1,kr2)

    elif encoding == 'alibi':
        heads = 12
        maxpos =  10
        batch_size = 1
        def build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
            #adpated from https://github.com/bigscience-workshop/bigscience/
            # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
            """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""
            def get_slopes(n):
                def get_slopes_power_of_2(n):
                    start = (2 ** (-2 ** -(math.log2(n) - 3)))
                    ratio = start
                    return [start * ratio ** i for i in range(n)]

                if math.log2(n).is_integer():
                    return get_slopes_power_of_2(n)
                else:
                    closest_power_of_2 = 2 ** math.floor(math.log2(n))
                    return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                        :n - closest_power_of_2]
            slopes = torch.Tensor(get_slopes(num_attention_heads))
            alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(num_attention_heads, -1, -1)
            alibi = alibi.repeat(batch_size, 1, 1)
            return alibi
        alibi1 = create_alibi_tensor(heads,maxpos)
        alibi2 = build_alibi_tensor(maxpos, heads, batch_size)
        torch.testing.assert_close(alibi1,alibi2)
        









