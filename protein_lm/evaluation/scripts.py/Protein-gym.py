######################################################################################################
# Script: ProteinGym Eval script
# Authors: Maximilian Sprang, Muedi
# Date: 08/2023
# Description: This script downloads and handles the zipped Protein Gym Csvs and preprocesses them 
# to be able to tokenized by EMS/ProtBERT tokenizers.
# Tokenization is done and then the esm 630M Model is used to be finetuned on ProteinGyms data
# ATM only substitution data is implemented for the finetunning but both are preprocessed and the
# complete datasets saved as CSV.
# finetuning is done with the evaluaten libray, which we'll likely change to an own trainling loop
# to be more flexible with our own models.  
######################################################################################################
# %%
import os
# funtiion to check if cwd is correct
def set_to_base_directory(repo_name):
    # Get the current working directory
    current_dir = os.getcwd()

    # Check if the repo name is in the path
    if repo_name in current_dir.split(os.sep)[-1]:
        print("Current directory is already the base directory of the repo.")
    else:
        print("Setting current directory as the base directory of the repo...")
        
        # Search for the repo name in the path
        base_dir = None
        path_parts = current_dir.split(os.sep)
        for i in range(len(path_parts) - 1, -1, -1):
            if path_parts[i] == repo_name:
                base_dir = os.sep.join(path_parts[:i+1])
                break
        
        if base_dir:
            os.chdir(base_dir)
            print("Changed to base directory:", base_dir)
        else:
            print("Repo name not found in the path.")

# check if cwd is base folder of repo
# needs to be moved intop main later
repo_name = "protein-lm-scaling"
set_to_base_directory(repo_name)

# huggingface
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from evaluate import load

from protein_lm.tokenizer import EsmTokenizer, AptTokenizer

# others
# import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
# http requests
import requests, zipfile, io, os

# %%
############################################## Functions ################################################ 

# this function takes the first row of a given DMS dataframe and corrects the given mutant
# in the given sequence to get the base sequence
def get_base_sequence(df: pd.DataFrame):
    mutations = df.loc[0, "mutant"].split(':')
    seq = df.loc[0, "mutated_sequence"]
    for mutation in mutations:
        base_AA = mutation[0]
        # mut_AA = mutation[-1]
        position = int(mutation[1:-1])
        # index seq at given position - 1 as the AA index doesn't start at 0
        base_seq = seq[:position - 1] + base_AA + seq[position:] 
        seq = base_seq
    return base_seq

def label_seq(seq, base_seq, ):
    # this function computes the score for a given sequence and base sequence
    # the way to go is to mask the token, that we know is mutated and let the model fill in the mask.
    # then compute the score by substracting the topken probabilities of the WT with the MT probs
    pass

# %%
# download substitutions, unzip, save to disk
path = "data/ProteinGym/"
sub_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip"

if os.path.exists(path + "ProteinGym_substitutions"):
    print("substitution data is already here :)")
else:
    print("download substitution data ...")
    r = requests.get(sub_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

# download indels, unzip, save to disk
sub_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_indels.zip"

if os.path.exists(path + "ProteinGym_indels"):
    print("indel data is already here :)")
else:
    print("download indel data ...")
    r = requests.get(sub_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

# %%
# load substitution data, introduce whitespeces and CLS/EOS tokens
# save complete data as csv, load as HF dataset
if os.path.exists(path + "ProteinGym_substitutions.csv") and os.path.exists(path + "ProteinGym_substitutions_base_seqs.csv"):
    print("preprocessing was already done, load csv")
    dataset = load_dataset("csv", data_files=(path + "ProteinGym_substitutions.csv"))
    base_seqs = pd.read_csv(path + "ProteinGym_substitutions_base_seqs.csv")
else:
    print("preprocess substitutions ...")
    folder_path = "data/ProteinGym/ProteinGym_substitutions"
    all_data = []
    base_seqs = {} # init dict to save baseseqs
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            experiment = filename[:-4]
            # save base_seqs, as experiment:sequence pair, needs to be before changing mutant name :)
            base_seqs[experiment] = get_base_sequence(df)
            # add experiment name to track and get base sequence for zero-shot tasks
            df["mutant"] = experiment + "_" + df["mutant"]
            all_data.append(df)
            
    # get dataframe
    merged_data = pd.concat(all_data, ignore_index=True)
    base_seqs = pd.DataFrame(base_seqs, index=["base_seq"]).T
    base_seqs = base_seqs.reset_index(names="experiment")
    # save the baseseqs
    base_seqs.to_csv(path + "ProteinGym_substitutions_base_seqs.csv", index=False)
    # Add spaces between each amino acid in the 'mutated_sequences' column
    # merged_data['mutated_sequence'] = merged_data['mutated_sequence'].apply(lambda seq: ' '.join(list(seq)))
    # add cls and end tokens
    merged_data['mutated_sequence'] = "<cls>" + merged_data['mutated_sequence'] + "<eos>"
    # save csv
    merged_data.to_csv(path + "ProteinGym_substitutions.csv", index=False)
    dataset = load_dataset("csv", data_files=(path + "ProteinGym_substitutions.csv"))
    del merged_data

# %% tokenize, with esm2_t33_650M_UR50D, use same checkpoint for model
checkpoint = "facebook/esm2_t33_650M_UR50D"
# autoTokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer = AptTokenizer()

def tokenize(batch):
    tokens = tokenizer(batch["mutated_sequence"], return_tensors=True, max_length=760)
    return {"input_ids": tokens}

token_data = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# rename and remove stuff to fit into dataloader seemlessly
# removes info we don't use in the network, as we only use tokens and binned scores
token_data = token_data.remove_columns(["DMS_score", "mutant", "mutated_sequence"])
# binned scores are renamed to 'labels'
token_data = token_data.rename_column("DMS_score_bin", "labels")

# Split the train dataset into train, valid, and test subsets
dict_train_test = token_data['train'].train_test_split(test_size=0.4, shuffle=True)
train_dataset = dict_train_test['train']
test_dataset = dict_train_test['test']

# subset for testruns:
# train_dataset = train_dataset.select([x for x in range(200)])
# test_dataset = test_dataset.select([x for x in range(100)])

# # here we could split into validation and test if needed
# dict_test_valid = test_dataset.train_test_split(test_size=0.5, shuffle=True)
# test_dataset = dict_test_valid['test']
# valid_dataset = dict_test_valid['train']
# %%  taken from facebooks pretrained-finetuning notebook here: 
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb#scrollTo=fc164b49
supervised=False
if supervised:
    # load model for seq classification
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

    model_name = checkpoint.split("/")[-1]
    batch_size = 8

    args = TrainingArguments(
        f"{model_name}-finetuned-localization",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    metric = load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # run trainer, this will return eval loass andd accuracy every few steps
    # and save this to the disk in the esm2* folder
    trainer.train()
else:
    # here we'll add a zero-shot eval script like this: 
    # https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py
    pass