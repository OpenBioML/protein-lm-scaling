from typing import Dict, Literal

from pydantic import BaseModel
from transformers import default_data_collator
from protein_lm.utils.fim_collator import FimDataCollator

class DataCollatorConfig(BaseModel):
    data_collator_type: Literal["default","fim"]


def get_data_collator(config_dict: Dict):
    config = DataCollatorConfig(**config_dict)
    if config.data_collator_type == "default":
      return default_data_collator
    elif config.data_collator_type=="fim":
      return FimDataCollator()
    else: 
      raise ValueError(f"Invalid data_collator_type {config.data_collator_type}")
