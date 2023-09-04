import math
import torch

def get_slopes(n):
    """
    Function to compute the m constant for each attention head. Code has been adapted from the official ALiBi codebase at:
    https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n))   
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

 
def create_alibi_tensor(attn_heads,maxpos):
    slopes = torch.Tensor(get_slopes(attn_heads))
    #The softmax operation is invariant to translation, and bias functions used are always linear. 
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
    return alibi.view(attn_heads, 1, maxpos)
    

