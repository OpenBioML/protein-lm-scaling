from transformers import DataCollatorForLanguageModeling
from protein_lm.tokenizer.tokenizer import AptTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import torch
import random
from collections.abc import Mapping
from transformers import BertTokenizer
import numpy as np

class FimDataCollator(DataCollatorForLanguageModeling):
    def __init__(self,
        tokenizer=None,     
        min_span_len: int = 1, 
        max_span_len: int = 10,
        fim_freq: float = 0.5, 
        mlm_probability=0.15,
        mlm=False,
        **kwargs ):

        if tokenizer is None:
            self.tokenizer=AptTokenizer()
        else:
            self.tokenizer=tokenizer

        super().__init__(tokenizer=self.tokenizer,mlm_probability=mlm_probability, **kwargs)
        self.fim_freq = fim_freq
        self.min_span_len=  min_span_len
        self.max_span_len= max_span_len
        self.mlm=mlm

    def fim_transform(self, inputs):
        assert self.min_span_len <= self.max_span_len, "min_span_len cannot be larger than max_span_len"
        if inputs.shape[0] < 5:
            return inputs
        max_span_len = min(self.max_span_len, inputs.shape[0])

        if self.min_span_len == max_span_len:
            span_len=self.min_span_len
        else:
            span_len=random.randint(self.min_span_len, max_span_len)
        span_start=random.randint(0,inputs.shape[0]-span_len)
        span_end=span_start + span_len
        middle_span_token = torch.tensor([self.tokenizer.middle_span_id], dtype=torch.long, device=inputs.device)
        end_span_token = torch.tensor([self.tokenizer.end_span_id], dtype=torch.long, device=inputs.device)
        
        fim_transformation = [                 
            inputs[ :span_start],             # <cls>+ Tokens before the masked span
            middle_span_token,                # <middle_span>
            inputs[ span_end:],               # Tokens after the masked span + <eos>
            inputs[ span_start:span_end], #tokens in middle
            end_span_token                      # <end_span>
        ]

        inputs=torch.cat(fim_transformation)

        return inputs
  
    
    def torch_call(self, examples)-> Dict[str, Any]:
        # Override the torch_call method to make custom fim collator.
        tokenize = BertTokenizer.from_pretrained("bert-base-uncased")
        if isinstance(examples[0], Mapping):
            batch = tokenize.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:

            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:

            if random.random() > self.fim_freq:
                #print("start",batch["input_ids"].shape)
                fim_batch= [self.fim_transform(inputs).unsqueeze(0) for inputs in batch['input_ids']]
                batch['input_ids']=torch.cat(fim_batch)
                attention_mask = torch.ones_like(batch["input_ids"])
                batch["attention_mask"] = attention_mask
                labels=batch['input_ids'].clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                batch["labels"] = labels
                #print(batch['input_ids'].shape,batch['labels'].shape,batch["attention_mask"].shape)

            else:
                labels = batch["input_ids"].clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                batch["labels"] = labels

        return batch
