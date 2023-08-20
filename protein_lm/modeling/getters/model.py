from typing import Any, Dict, Literal, Optional

import torch
from pydantic import BaseModel

from protein_lm.modeling.models.apt.config import APTConfig
from protein_lm.modeling.models.apt.model_pytorch import APTLMHeadModel


class ModelConfig(BaseModel):
    # If desired, this can be modified to support a variety of model types
    model_type: Literal["APT"]
    model_config_args: Dict
    pretrained_checkpoint: Optional[str]


def get_model(config_dict: Dict):
    config = ModelConfig(**config_dict)
    if config.model_type == "APT":
        model_constructor = APTLMHeadModel
        model_config_constructor = APTConfig
    else:
        raise ValueError(f"Invalid ModelConfig.model_type {config.model_type}")

    model_config = model_config_constructor(**config.model_config_args)
    if config.pretrained_checkpoint is None:
        model = model_constructor(config=model_config)
    else:
        model = model_constructor.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_checkpoint,
            config=model_config,
        )

    if torch.cuda.is_available():
        model.cuda()

    return model
