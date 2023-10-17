import os
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
strategy2col = {'ppl': 'ppl'} #mapping of strategy to the computed column name storing the values of respective strategy
total = 0 #number of batches/steps
unsorted = 0 #number of unsorted batches/steps
InputDataClass = NewType("InputDataClass", Any)
def cl_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    global total
    global unsorted
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
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
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                if k == 'ppl':
                    batch[k] = [-f[k] for f in features]
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    lens = batch['ppl']
    print('######lens(cl_data_collator)#########')
    print(lens)
    total = total + 1
    try:
        assert lens == sorted(lens)
    except:
        unsorted = unsorted + 1
        print('not sorted')
    return {'input_ids':batch['input_ids'],'labels': batch['labels']}


def create_random_dataframe(sequence_column_name = 'sequence',
                            curriculum_learning_column_name = 'ppl',
                            curriculum_learning_strategy = 'ppl',
                            max_sequence_length = 30,
                            max_perplexity = 100.0,
                            n = 5000):
  assert max_sequence_length > 2
  random.seed(42)
  df = pd.DataFrame()
  def create_sequence(length):
    seq = ''.join(random.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(length))
    return seq

  if curriculum_learning_strategy == 'ppl':
    df[sequence_column_name] = [create_sequence(random.randint(2, max_sequence_length)) for i in range(n)]
    df[curriculum_learning_column_name] = [random.uniform(1.0, max_perplexity) for _ in range(n)]
    return df

@pytest.mark.parametrize("strategy",strategies)
def test_curriculum_learning(strategy):
    
    with open(CONFIG_PATH, "r") as cf:
        print('loading file.....')
        config_dict = yaml.safe_load(cf)
        
        config_dict['dataset']['max_sequence_length'] = 40
        config_dict['dataset']['do_curriculum_learning'] = True
        config_dict['dataset']['curriculum_learning_column_name'] = strategy2col[strategy]
        config_dict['dataset']['curriculum_learning_strategy'] = strategy
        config_dict['dataset']['val_size'] = 100
        config_dict['dataset']['test_size'] = 100
        config_dict['dataset']['subsample_size'] = 500
        config_dict["training_arguments"]['group_by_length'] = True
        config_dict["training_arguments"]['length_column_name'] = config_dict['dataset']['curriculum_learning_column_name']
        config_dict["training_arguments"]['remove_unused_columns'] = False # this is necessary to keep curriculum_learning_column_name
        config_dict["training_arguments"]['per_device_train_batch_size'] = 20
        config_dict["training_arguments"]['max_steps'] = -1
        config_dict["training_arguments"]['num_train_epochs'] = 2

        print(config_dict)
    
    tokenizer = get_tokenizer(config_dict=config_dict["tokenizer"])
    dataset = DatasetDict()
    val_df = create_random_dataframe(sequence_column_name = config_dict['dataset']['sequence_column_name'],curriculum_learning_column_name = config_dict['dataset']['curriculum_learning_column_name'],max_sequence_length = config_dict['dataset']['max_sequence_length'], n = config_dict['dataset']['val_size'] )
    test_df = create_random_dataframe(sequence_column_name = config_dict['dataset']['sequence_column_name'],curriculum_learning_column_name = config_dict['dataset']["curriculum_learning_column_name"],max_sequence_length = config_dict['dataset']['max_sequence_length'], n = config_dict['dataset']['test_size'] )
    train_df = create_random_dataframe(sequence_column_name = config_dict['dataset']['sequence_column_name'],curriculum_learning_column_name = config_dict['dataset']["curriculum_learning_column_name"],max_sequence_length = config_dict['dataset']['max_sequence_length'], n = config_dict['dataset']['subsample_size'] )

    dataset['train'] = Dataset.from_pandas(train_df)
    dataset['val'] = Dataset.from_pandas(val_df)
    dataset['test'] = Dataset.from_pandas(test_df)
    dataset = dataset.map(
        lambda e: set_input_ids(
            result=e,
            tokenizer=tokenizer,
            sequence_column_name=config_dict['dataset']['sequence_column_name'],
            max_sequence_length=config_dict['dataset']['max_sequence_length'],
        ),
        batched=True,
    )
    dataset = dataset.map(set_labels, batched=True)
    dataset = dataset.map(lambda e: batch_set_curriculum_learning_column(
        result = e,
        input_column_name = config_dict['dataset']['sequence_column_name'],
        curriculum_learning_column_name = config_dict['dataset']['curriculum_learning_column_name'],
        strategy = config_dict['dataset']['curriculum_learning_strategy']

    ),batched=True)
    dataset = dataset.select_columns(['input_ids', 'labels', strategy2col[strategy]])
    model = get_model(
        config_dict=config_dict["model"],
    )
    
    training_args = get_training_args(
        config_dict=config_dict["training_arguments"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("val", None),
        data_collator=cl_data_collator,
    )
    
    trainer.train()
    percentage_unsorted = int((unsorted / total) * 100) #computing the number of times the list in collator was not sorted
    #there are sometimes cases where the list is off by a few entries aa the LengthGroupedSampler has a bit of randomness
    print(f'percentage_unsorted:{percentage_unsorted}')
    assert percentage_unsorted < 10 # just a rough heuristic