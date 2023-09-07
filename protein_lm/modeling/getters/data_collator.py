from typing import Dict, Literal

from pydantic import BaseModel
from transformers import default_data_collator


class DataCollatorConfig(BaseModel):
    data_collator_type: Literal["default"]


def get_data_collator(config_dict: Dict):
    config = DataCollatorConfig(**config_dict)
    if config.data_collator_type == "default":
        return default_data_collator
    else:
        raise ValueError(f"Invalid data_collator_type {config.data_collator_type}")
