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
        query_zero_init = True,
        n_layer = None,
        contact_prediction_head = False,
        initializer_range = 0.02,
        # whether to use MuParametrization
        use_mup = False, 
        # whether to initialize the output (readout) layer with zero-initialization
        readout_zero_init = True, 
        # the output layer multiplier if mup is used, see https://github.com/microsoft/mup/blob/19814971934ef91dd546f88e913fc963e096d11c/mup/layer.py#L56
        mup_output_mult = 1.0,
        width_mult_for_weights = 2.0,
        # rope
        rope_theta = 0.0,
        rope_scaling_factor=1,
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
        self.contact_prediction_head = contact_prediction_head
        self.initializer_range = initializer_range
        self.readout_zero_init = readout_zero_init
        self.mup_output_mult = mup_output_mult
        self.width_mult_for_weights = width_mult_for_weights
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor

