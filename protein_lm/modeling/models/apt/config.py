from transformers import GPT2Config


class APTConfig(GPT2Config):
    """
    Config subclass for Autoregressive Protein Transformer (APT) model architecture.
    """

    def __init__(
        self,
        attention_mode="APT",
        position_embedding="rotary",
        tokenizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nn_model_type = "APT"
        self.attention_mode = attention_mode
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
