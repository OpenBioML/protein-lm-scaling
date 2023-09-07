import os
from typing import Dict, List, Union

from pydantic import BaseModel, FieldValidationInfo, field_validator
from transformers import TrainingArguments


class TrainingArgsConfig(BaseModel):
    per_device_train_batch_size: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    max_steps: int
    save_steps: int
    output_dir: str
    save_strategy: str
    report_to: str
    label_names: List[str]
    no_cuda: bool

    @field_validator(
        "per_device_train_batch_size",
        "num_train_epochs",
        "weight_decay",
        "learning_rate",
        "save_steps",
    )
    @classmethod
    def check_gt_zero(cls, v: Union[int, float], info: FieldValidationInfo):
        if v <= 0:
            raise ValueError(f"trainer.{info.field_name} must be greater than 0")
        return v


def get_training_args(config_dict: Dict) -> TrainingArguments:
    config = TrainingArgsConfig(**config_dict)

    if not os.path.isdir(config.output_dir):
        print(f"creating checkpoint directory at {config.output_dir}")
        os.makedirs(config.output_dir)

    return TrainingArguments(
        **config_dict,
    )
