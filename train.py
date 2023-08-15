import argparse
import json
import pandas as pd
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import torch
from datasets import load_dataset,Dataset
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import DataCollatorForLanguageModeling
from protein_lm.modeling.models.apt.model_pytorch import APTLMHeadModel
from protein_lm.modeling.models.apt.config import APTConfig
from protein_lm.tokenizer.tokenizer import AptTokenizer

dir_path = os.path.dirname(os.path.abspath(__file__))

def set_input_ids(result= None ,tokenizer = None, sequence_column_name = 'sequence', max_sequence_length = 1024):
  result['input_ids'] = tokenizer.batch_encode(sequences=result[sequence_column_name],max_sequence_length=max_sequence_length)
  return result

def set_labels(result):
  result['labels'] = result['input_ids'].copy()
  return result


def train(**kwargs):
    """
    Main script to train APT.
    """
    mode = kwargs.get('mode','command_line')
    if mode == 'command_line':
      parser = argparse.ArgumentParser(description='APT Training')
      parser.add_argument('--model_name', default='gpt2',type=str, help='APT model name')
      parser.add_argument('--model_framework', default='pytorch', type=str, help='Underlying framework [pytorch|JAX]')
      parser.add_argument('--hf_dataset_name', default='zpn/uniref50', type=str, help='name of hugging face dataset ')
      parser.add_argument('--sequence_column_name', default='sequence', type=str, help='name of sequence column in dataset ')
      parser.add_argument('--max_sequence_length', default=1024, type=int, help='maximum length of sequence in dataset')
      parser.add_argument('--train_file_path', default='', type=str, help='Path to training file ')
      parser.add_argument('--train_sample_size', default=1000, type=str, help='number of training samples')
      parser.add_argument('--train_batch_size', default=16, type=int, help='Batch size for training')
      parser.add_argument('--epochs', default=1000, type=int, help='Batch size for training')
      parser.add_argument('--lr', default=2e-5, type=int, help='learning_rate for training')
      parser.add_argument('--wd', default=0.01, type=int, help='weight_decay for training')
      parser.add_argument('--steps_per_epoch', default=5, type=int, help='step per each training epoch')
      parser.add_argument('--model_checkpoint_path', type=str, help='Path of APT model checkpoint')
      parser.add_argument('--checkpoint_steps', default=500, type=int, help='steps per checkpoint save')
      parser.add_argument('--checkpoint_dir',default=os.path.join(dir_path,'protein_lm','checkpoints'), type=str, help='Path of APT model checkpoints directory')

      args = parser.parse_args()
      model_name = args.model_name
      model_framework = args.model_framework
      train_file_path = args.train_file_path
      train_sample_size = args.train_sample_size
      train_batch_size = args.train_batch_size
      hf_dataset_name = args.hf_dataset_name
      sequence_column_name = args.sequence_column_name
      max_sequence_length = args.max_sequence_length
      epochs = args.epochs
      steps = args.steps_per_epoch
      lr = args.lr
      wd = args.wd
      checkpoint_steps = args.checkpoint_steps
      checkpoint_path = args.model_checkpoint_path
      checkpoint_dir = args.checkpoint_dir


    else:
      model_name = kwargs.get('model_name','gpt2')
      model_framework = kwargs.get('model_framework','pytorch')
      train_file_path = kwargs.get('train_file_path','')
      train_sample_size = kwargs.get('train_sample_size',1000)
      train_batch_size = kwargs.get('train_batch_size',16)
      hf_dataset_name = kwargs.get('hf_dataset_name','zpn/uniref50')
      sequence_column_name =  kwargs.get('sequence_column_name','sequence')
      max_sequence_length = kwargs.get('max_sequence_length',1024)
      epochs = kwargs.get('epochs',100)
      steps = kwargs.get('steps_per_epoch',100)
      lr = kwargs.get('lr',2e-5)
      wd = kwargs.get('wd',0.01)
      checkpoint_path = kwargs.get('model_checkpoint_path','')
      checkpoint_steps = kwargs.get('checkpoint_steps',500)
      checkpoint_dir = kwargs.get('checkpoint_dir',os.path.join(dir_path,'protein_lm','checkpoints'))
    
    if not os.path.isdir(checkpoint_dir):
      print(f'creating checkpoint directory at {checkpoint_dir}')
      os.makedirs(checkpoint_dir)

    assert model_name == 'gpt2'
    assert model_framework == 'pytorch'
    assert os.path.isfile(train_file_path) or (isinstance(hf_dataset_name,str) and len(hf_dataset_name) > 0)
    assert train_batch_size > 0
    assert train_sample_size > 0
    assert max_sequence_length > 0
    assert lr > 0 
    assert wd > 0
    assert epochs > 0 
    assert steps > 0 
    assert checkpoint_steps > 0
    assert (isinstance(sequence_column_name,str) and len(sequence_column_name) > 0)

    tokenizer = AptTokenizer()

    if train_file_path:
      train_df = pd.read_csv(train_file_path).head(train_sample_size)
      train_ds = Dataset.from_pandas(train_df)
      train_ds = train_ds.map(lambda e: set_input_ids(result =e,tokenizer= tokenizer, sequence_column_name = sequence_column_name , max_sequence_length = max_sequence_length), batched=True)
      train_ds = train_ds.map(set_labels, batched=True)


    elif hf_dataset_name:
      train_ds = load_dataset(hf_dataset_name,streaming=True,split='train')
      train_ds = train_ds.take(train_sample_size)
      train_ds = train_ds.map(lambda e: set_input_ids(result =e,tokenizer= tokenizer, sequence_column_name = sequence_column_name , max_sequence_length = max_sequence_length), batched=True)
      train_ds = train_ds.map(set_labels, batched=True)
      train_ds = train_ds.with_format("torch")


    config = APTConfig()
    config.attention_mode="APT"
    if model_framework=="pytorch":
        model = APTLMHeadModel.from_pretrained(pretrained_model_name_or_path=model_name,config=config)
        if torch.cuda.is_available():
            model.cuda()

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        max_steps=steps,
        num_train_epochs=epochs,
        label_names= ['labels'],
        per_device_train_batch_size=train_batch_size,
        learning_rate=lr,
        weight_decay=wd,
        save_strategy = "epoch",
        save_steps = checkpoint_steps,
        report_to="none",
        no_cuda=False,
    )
    model.train()
    trainer = Trainer(
            model=model,
            args = training_args,
            train_dataset=train_ds,
            data_collator=default_data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics
    try:
        perplexity = math.exp(metrics["train_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    print('metrics:',metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == '__main__':
  train()
