import os
from typing import Dict, Union

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

    @field_validator(
        "per_device_train_batch_size",
        "num_train_epochs",
        "max_steps",
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

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        report_to=config.report_to,
        label_names=["labels"],
        no_cuda=False,
    )
    return training_args
