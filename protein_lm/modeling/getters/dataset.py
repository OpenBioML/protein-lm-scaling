from typing import Dict, Literal, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    dataset_type: Literal["csv", "huggingface"]

    # The path if local or the huggingface dataset name if huggingface
    dataset_loc: str

    # train sample size to limit to, if any
    train_sample_size: Optional[int] = None

    # name of the column that contains the sequence
    sequence_column_name: str

    max_sequence_length: int


def set_input_ids(
    result=None,
    tokenizer=None,
    sequence_column_name="sequence",
    max_sequence_length=1024,
):
    result["input_ids"] = tokenizer(
        result[sequence_column_name],
        max_sequence_length=max_sequence_length,
        add_special_tokens=True,
        return_tensors=True,
    )
    return result


def set_labels(result):
    result["labels"] = result["input_ids"].copy()
    return result


def get_local_dataset(config: DatasetConfig) -> Dataset:
    train_ds = load_dataset("csv", data_files=config.dataset_loc)["train"]
    return train_ds


def get_huggingface_dataset(config: DatasetConfig) -> Dataset:
    train_ds = load_dataset(config.dataset_loc, streaming=True, split="train")
    return train_ds


def get_dataset(config_dict: Dict, tokenizer) -> Dataset:
    config = DatasetConfig(**config_dict)
    if config.dataset_type == "csv":
        train_ds = get_local_dataset(config)
    elif config.dataset_type == "huggingface":
        train_ds = get_huggingface_dataset(config)
    else:
        raise ValueError(f"Invalid dataset_type {config.dataset_type}!")

    train_ds = train_ds.map(
        lambda e: set_input_ids(
            result=e,
            tokenizer=tokenizer,
            sequence_column_name=config.sequence_column_name,
            max_sequence_length=config.max_sequence_length,
        ),
        batched=True,
    )
    train_ds = train_ds.map(set_labels, batched=True)
    return train_ds
