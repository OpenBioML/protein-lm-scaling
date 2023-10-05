import os
from transformers import PreTrainedTokenizerFast
from transformers import EsmTokenizer as EsmTokenizerBase

esm_path = os.path.join(os.path.dirname(__file__), 'esm_tokenizer')
apt_path = os.path.join(os.path.dirname(__file__), 'apt_tokenizer')


def EsmTokenizer():
    return EsmTokenizerBase.from_pretrained(esm_path)

def AptTokenizer():
    return PreTrainedTokenizerFast.from_pretrained(apt_path)
