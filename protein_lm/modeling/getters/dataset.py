from typing import Dict, Literal, Optional

from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    dataset_type: Literal["csv", "huggingface"]

    # The path if local or the huggingface dataset name if huggingface
    dataset_loc: str

    # sample size to limit to, if any, usually for debugging
    subsample_size: Optional[int] = None

    """
    Args for splitting into train, val, test
    to be updated once we have more options
    """
    # split seed
    split_seed: Optional[int] = None
    # size of validation dataset
    val_size: int
    # size of test dataset
    test_size: int

    # name of the column that contains the sequence
    sequence_column_name: str
    
    max_sequence_length: int
    do_curriculum_learning: bool
    curriculum_learning_strategy: str
    curriculum_learning_column_name: str


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

def batch_set_curriculum_learning_column(
    result=None,
    input_column_name='sequence',
    curriculum_learning_column_name='sequence_length',
    strategy='sequence_length'
):
    supported_strategies = ['sequence_length', 'ppl', 'plddt']

    if strategy not in supported_strategies:
        raise Exception(f'Invalid {strategy} provided. Supported strategy values include {", ".join(supported_strategies)}')

    if strategy == 'sequence_length':
        # LengthGroupedSampler sorts in descending so we make it ascending by multiplying with -1
        result[curriculum_learning_column_name] = [-len(x) for x in result[input_column_name]]
    elif strategy in ['ppl', 'plddt']:
        # Assuming that the precomputed categories for 'ppl' and 'plddt' are stored in fields named 'ppl_category' and 'plddt_category'
        result[curriculum_learning_column_name] = [-x for x in result[strategy + "_category"]]

    return result

def set_labels(result):
    result["labels"] = result["input_ids"].copy()
    return result


def train_val_test_split(
    dataset_dict: DatasetDict,
    config: DatasetConfig,
) -> DatasetDict:
    """
    Given a dictionary of datasets that only contains the split "train",
    optionally subsamples it, and then splits it
    so that it has potentially 3 splits: "train", "val", "test", where
    "val" and "test" splits do not exist if the specified sizes are 0
    """
    assert set(dataset_dict.keys()) == {
        "train"
    }, f"{train_val_test_split.__name__} expects its input to have the keys \
        ['train'] but the input has keys {list(dataset_dict.keys())}"

    dataset = dataset_dict["train"]

    val_size = config.val_size
    test_size = config.test_size

    assert isinstance(
        dataset, Dataset
    ), f"Invalid dataset type {type(dataset)}, only datasets.Dataset allowed"

    dataset = dataset.shuffle(seed=config.split_seed)

    if config.subsample_size is not None:
        dataset = dataset.select(range(config.subsample_size))

    valtest_size = val_size + test_size

    if valtest_size > 0:
        train_valtest = dataset.train_test_split(
            test_size=val_size + test_size,
            shuffle=False,
        )
        split_dict = {
            "train": train_valtest["train"],
        }
        if test_size > 0 and val_size > 0:
            test_val = train_valtest["test"].train_test_split(
                test_size=test_size,
                shuffle=False,
            )
            split_dict["val"] = test_val["train"]
            split_dict["test"] = test_val["test"]
        elif val_size > 0:
            split_dict["val"] = train_valtest["test"]
        else:
            split_dict["test"] = train_valtest["test"]
    else:
        split_dict = {
            "train": dataset,
        }

    split_dataset_dict = DatasetDict(split_dict)
    return split_dataset_dict


def get_csv_dataset(config: DatasetConfig) -> Dataset:
    # note that a csv is read as having just one split "train"
    dataset_dict = load_dataset("csv", data_files=config.dataset_loc)
    return train_val_test_split(dataset_dict, config)


def get_huggingface_dataset(config: DatasetConfig) -> Dataset:
    dataset_dict = load_dataset(config.dataset_loc)
    if set(dataset_dict.keys()) == {"train", "val", "test"}:
        return dataset_dict

    assert set(dataset_dict.keys()) == {
        "train"
    }, f"Huggingface DatasetDicts should have the keys {{'train'}} or \
        {{'train', 'val', 'split'}} but this DatasetDict has keys \
            {set(dataset_dict.keys())}"
    return train_val_test_split(dataset_dict, config)


def get_dataset(config_dict: Dict, tokenizer) -> Dataset:
    config = DatasetConfig(**config_dict)

    if config.dataset_type == "csv":
        train_ds = get_csv_dataset(config)
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
    if config.do_curriculum_learning:
        train_ds = train_ds.map(lambda e: batch_set_curriculum_learning_column(
            result = e,
            input_column_name = config.sequence_column_name,
            curriculum_learning_column_name = config.curriculum_learning_column_name,
            strategy = config.curriculum_learning_strategy

        ),batched=True)

    return train_ds
