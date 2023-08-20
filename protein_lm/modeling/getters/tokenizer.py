from typing import Dict, Literal

from pydantic import BaseModel

from protein_lm.tokenizer.tokenizer import AptTokenizer


class TokenizerConfig(BaseModel):
    tokenizer_type: Literal["APT"]


def get_tokenizer(config_dict: Dict):
    config = TokenizerConfig(**config_dict)
    if config.tokenizer_type == "APT":
        tokenizer_constructor = AptTokenizer
    else:
        raise ValueError(f"Invalid tokenizer_type {config.tokenizer_type}")

    return tokenizer_constructor()
