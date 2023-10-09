import os
import sys
import pytest
import torch
import torch.nn as nn
import yaml
from transformers import Trainer
from protein_lm.modeling.getters.data_collator import get_data_collator
from protein_lm.modeling.getters.model import get_model
from protein_lm.modeling.getters.tokenizer import get_tokenizer
from protein_lm.modeling.getters.training_args import get_training_args
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from pydantic import BaseModel
from protein_lm.modeling.getters.dataset import DatasetConfig,get_csv_dataset,set_input_ids,set_labels,batch_set_curriculum_learning_column
##data collator imports
from dataclasses import dataclass
from typing import Dict, Literal,Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
import pandas as pd
import random


CONFIG_PATH = "protein_lm/configs/train/toy_localcsv.yaml"
strategies = ['ppl']
strategy2col = {'ppl': 'ppl_category'}

total = 0  # number of batches/steps
unsorted = 0  # number of unsorted batches/steps
InputDataClass = NewType("InputDataClass", Any)

def cl_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    global total
    global unsorted

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    lens = batch['ppl_category']
    total += 1
    if lens != sorted(lens):
        unsorted += 1

    return {'input_ids': batch['input_ids'], 'labels': batch['labels']}

def create_random_dataframe(sequence_column_name='sequence',
                            curriculum_learning_column_name='ppl_category',
                            curriculum_learning_strategy='ppl',
                            max_sequence_length=30, n=5000):

    assert max_sequence_length > 2
    random.seed(42)
    df = pd.DataFrame()

    def create_sequence(length):
        return ''.join(random.choice(['A', 'T', 'G', 'C']) for _ in range(length))

    df[sequence_column_name] = [create_sequence(random.randint(2, max_sequence_length)) for i in range(n)]
    df[curriculum_learning_strategy] = [random.uniform(0, 100) for _ in range(n)]
    categories = precompute_categories(df[curriculum_learning_strategy].tolist())
    df[curriculum_learning_column_name] = categories

    return df

def precompute_categories(values, levels=10):
    sorted_values = sorted(values)
    n = len(sorted_values)
    bins = [int(n * i / levels) for i in range(levels + 1)]
    boundaries = [sorted_values[i] for i in bins]
    categories = []
    for value in values:
        for i, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if low <= value < high:
                categories.append(-i - 1)
                break

    return categories

@pytest.mark.parametrize("strategy", strategies)
def test_curriculum_learning(strategy):
    with open(CONFIG_PATH, "r") as cf:
        config_dict = yaml.safe_load(cf)
        
        config_dict['dataset']['do_curriculum_learning'] = True
        config_dict['dataset']['curriculum_learning_column_name'] = strategy2col[strategy]
        config_dict['dataset']['curriculum_learning_strategy'] = strategy
        config_dict['dataset']['val_size'] = 100
        config_dict['dataset']['test_size'] = 100
        config_dict['dataset']['subsample_size'] = 500
        config_dict["training_arguments"]['group_by_length'] = True
        config_dict["training_arguments"]['length_column_name'] = config_dict['dataset']['curriculum_learning_column_name']
        config_dict["training_arguments"]['remove_unused_columns'] = False
        config_dict["training_arguments"]['per_device_train_batch_size'] = 20
        config_dict["training_arguments"]['max_steps'] = -1
        config_dict["training_arguments"]['num_train_epochs'] = 2
    
    # Assuming necessary functions for model, tokenizer, etc. are correctly imported.
    tokenizer = get_tokenizer(config_dict=config_dict["tokenizer"])
    model = get_model(config_dict=config_dict["model"])
    training_args = get_training_args(config_dict=config_dict["training_arguments"])
    
    dataset = DatasetDict()
    train_df = create_random_dataframe()
    val_df = create_random_dataframe(n=config_dict['dataset']['val_size'])
    test_df = create_random_dataframe(n=config_dict['dataset']['test_size'])

    dataset['train'] = Dataset.from_pandas(train_df)
    dataset['val'] = Dataset.from_pandas(val_df)
    dataset['test'] = Dataset.from_pandas(test_df)
    
    # Assuming necessary transformations and functions are correctly imported.
    dataset = dataset.map(set_input_ids, batched=True)
    dataset = dataset.map(set_labels, batched=True)
    dataset = dataset.map(lambda e: batch_set_curriculum_learning_column(e, strategy=strategy), batched=True)
    dataset = dataset.select_columns(['input_ids', 'labels', strategy2col[strategy]])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("val", None),
        data_collator=cl_data_collator
    )
    
    trainer.train()
    percentage_unsorted = int((unsorted / total) * 100)
    assert percentage_unsorted < 10

# To run the test:
if __name__ == "__main__":
    pytest.main()
