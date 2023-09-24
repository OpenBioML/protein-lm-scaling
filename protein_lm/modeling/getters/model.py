from typing import Dict, Literal, Optional

import torch
from pydantic import BaseModel

from protein_lm.modeling.models.apt.config import APTConfig
from protein_lm.modeling.models.apt.model_pytorch import APTLMHeadModel


class NNModelConfig(BaseModel):
    # If desired, this can be modified to support a variety of model types
    # Note: "nn_model_.." because anything with the "model_" prefix leads to
    # pydantic namespace warnings
    nn_model_type: Literal["APT"]
    nn_model_config_args: Dict
    pretrained_checkpoint: Optional[str]


def get_model(config_dict: Dict):
    config = NNModelConfig(**config_dict)
    if config.nn_model_type == "APT":
        model_constructor = APTLMHeadModel
        model_config_constructor = APTConfig
    else:
        raise ValueError(f"Invalid NNModelConfig.nn_model_type {config.nn_model_type}")

    model_config = model_config_constructor(**config.nn_model_config_args)
    if config.pretrained_checkpoint is None:
        model = model_constructor(config=model_config)
    else:
        model = model_constructor.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_checkpoint,
            config=model_config,
        )

    return model
