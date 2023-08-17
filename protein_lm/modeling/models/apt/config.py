from transformers import GPT2Config

class APTConfig(GPT2Config):
    """
    Config subclass for Autoregressive Protein Transformer (APT) model architecture.
    """
    def __init__(
        self,
        tokenizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_type="APT"
        self.tokenizer = tokenizer
