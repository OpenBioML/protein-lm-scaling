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
        use_mup = False,
        query_zero_init = True,
        n_layer = None,
        initializer_range = 0.02,
        mup_init_scale = 1.0,
        mup_output_mult = 1.0,
        mup_attn_mult = 1.0,
        mup_embedding_mult = 1.0,
        mup_rp_embedding_mult = 1.0,
        mup_width_scale = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nn_model_type = "APT"
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        
        self.use_mup = use_mup
        self.query_zero_init = query_zero_init,
        self.n_layer = n_layer
        self.initializer_range = initializer_range
        self.mup_init_scale = mup_init_scale
        self.mup_output_mult = mup_output_mult
        self.mup_attn_mult = mup_attn_mult
        self.mup_embedding_mult = mup_embedding_mult
        self.mup_rp_embedding_mult = mup_rp_embedding_mult
        self.mup_width_scale = mup_width_scale

