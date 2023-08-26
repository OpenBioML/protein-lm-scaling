import math

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

 