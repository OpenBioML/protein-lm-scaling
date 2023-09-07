from transformers import GPT2Config


class APTConfig(GPT2Config):
    """
    Config subclass for Autoregressive Protein Transformer (APT) model architecture.
    """

    def __init__(
        self,
        position_embedding="learned",
        tokenizer=None,
        max_sequence_length = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nn_model_type = "APT"
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

