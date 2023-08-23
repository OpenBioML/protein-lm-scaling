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
#####################################################################
# SOME VARIABLES    
# relative protgym path
path = "protein_lm/dataset/ProteinGym/"
# run supervised benchmark
supervised=False
# run zero shot benchmark without gpu
nogpu = False
# scoring strategy for zero shot"
# choose out of: ["wt-marginals", "masked-marginals", "pseudo-ppl"]
scoring_strategy = 'wt-marginals'
# relative output path 
dms_output =  "protein_lm/evaluation/output/{}".format(scoring_strategy)
# %%
# download substitutions, unzip, save to disk
dat_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip"

if os.path.exists(path + "ProteinGym_substitutions"):
    print("substitution data is already here :)")
else:
    print("download substitution data ...")
    r = requests.get(dat_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

# download indels, unzip, save to disk
dat_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_indels.zip"

if os.path.exists(path + "ProteinGym_indels"):
    print("indel data is already here :)")
else:
    print("download indel data ...")
    r = requests.get(dat_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

# download ref files
dat_url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/ProteinGym_reference_file_substitutions.csv"
if os.path.exists(path + "ProteinGym_reference_file_substitutions.csv"):
    print("Substitution reference file is already here :)")
else:
    print("download substitution reference ...")
    r = requests.get(dat_url)
    df = pd.read_csv(io.BytesIO(r.content))
    df.to_csv(path + "ProteinGym_reference_file_substitutions.csv", index=False)

dat_url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/ProteinGym_reference_file_indels.csvv"
if os.path.exists(path + "ProteinGym_reference_file_indels.csv"):
    print("Indel reference file is already here :)")
else:
    print("download Indel reference ...")
    r = requests.get(dat_url)
    df = pd.read_csv(io.BytesIO(r.content))
    df.to_csv(path + "ProteinGym_reference_file_indels.csv", index=False)
# %%
# load substitution data, introduce whitespeces and CLS/EOS tokens
# save complete data as csv, load as HF dataset
if os.path.exists(path + "ProteinGym_substitutions.csv"):
    print("preprocessing was already done, load csv")
    dataset = load_dataset("csv", data_files=(path + "ProteinGym_substitutions.csv"))
else:
    print("preprocess substitutions ...")
    folder_path = "data/ProteinGym/ProteinGym_substitutions"
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            experiment = filename[:-4]
    
    
            # add experiment name to track and get base sequence for zero-shot tasks
            df["mutant"] = experiment + "_" + df["mutant"]
            all_data.append(df)
            
    # get dataframe
    merged_data = pd.concat(all_data, ignore_index=True)
    # save the baseseqs
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
    tokens = tokenizer(batch["mutated_sequence"], return_tensors=True, max_sequence_length=760)
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
        
    import torch

    from protein_lm.modeling.models.esm.data import Alphabet, FastaBatchedDataset
    from protein_lm.modeling.models.esm import pretrained
    from protein_lm.modeling.models.esm.model.msa_transformer import MSATransformer
    import pandas as pd
    from tqdm import tqdm
    from Bio import SeqIO
    import itertools
    from typing import List, Tuple
    import numpy as np


    def remove_insertions(sequence: str) -> str:
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        # This is an efficient way to delete lowercase characters and insertion characters from a string
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None

        translation = str.maketrans(deletekeys)
        return sequence.translate(translation)


    def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
        
        The input file must be in a3m format (although we use the SeqIO fasta parser)
        for remove_insertions to work properly."""

        msa = [
            (record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
        ]
        return msa


    def label_row(row, sequence, token_probs, alphabet, offset_idx):
        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        return score.item()


    def compute_pppl(row, sequence, model, alphabet, offset_idx):
        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        # modify the sequence
        sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

        # encode the sequence
        data = [
            ("protein1", sequence),
        ]

        batch_converter = alphabet.get_batch_converter()

        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # compute probabilities at each position
        log_probs = []
        for i in range(1, len(sequence) - 1):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
            log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
        return sum(log_probs)
    
    # get experiments and base seqs
    ref_df = pd.read_csv(path + "ProteinGym_reference_file_substitutions.csv")
    dms_ids = ref_df.DMS_id
    dms_file = ref_df.DMS_filename
    dms_ref_seqs = ref_df.target_seq

    for experiment, file_name, sequence in zip(dms_ids, dms_file, dms_ref_seqs):

        # Load the deep mutational scan
        dms_df_path = path + "ProteinGym_substitutions/" + file_name
        dms_df = pd.read_csv(dms_df_path)
        
        # inference for each model
        # set checkpoint to be mnodel location for now
        model_location = checkpoint.split("/")[-1]

        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        if torch.cuda.is_available() and not nogpu:
            model = model.cuda()
            print("Transferred model to GPU")

        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):
            # as far as I know we do not plan on using this? I kept it around for now.
            pass
            # data = [read_msa(args.msa_path, args.msa_samples)]
            # assert (
            #     scoring_strategy == "masked-marginals"
            # ), "MSA Transformer only supports masked marginal strategy"

            # batch_labels, batch_strs, batch_tokens = batch_converter(data)

            # all_token_probs = []
            # for i in tqdm(range(batch_tokens.size(2))):
            #     batch_tokens_masked = batch_tokens.clone()
            #     batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
            #     with torch.no_grad():
            #         token_probs = torch.log_softmax(
            #             model(batch_tokens_masked.cuda())["logits"], dim=-1
            #         )
            #     all_token_probs.append(token_probs[:, 0, i])  # vocab size
            # token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            # dms_df[model_location + "_" + experiment] = dms_df.apply(
            #     lambda row: label_row(
            #         row[args.mutation_col], sequence, token_probs, alphabet, args.offset_idx
            #     ),
            #     axis=1,
            # )

        else:
            data = [
                ("protein1", sequence),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            if scoring_strategy == "wt-marginals":
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)
                dms_df[model_location + "_" + experiment] = dms_df.apply(
                    lambda row: label_row(
                        row[args.mutation_col],
                        sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif scoring_strategy == "masked-marginals":
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1))):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    with torch.no_grad():
                        token_probs = torch.log_softmax(
                            model(batch_tokens_masked.cuda())["logits"], dim=-1
                        )
                    all_token_probs.append(token_probs[:, i])  # vocab size
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                dms_df[model_location + "_" + experiment] = dms_df.apply(
                    lambda row: label_row(
                        row[args.mutation_col],
                        sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                dms_df[model_location + "_" + experiment] = dms_df.progress_apply(
                    lambda row: compute_pppl(
                        row[args.mutation_col], sequence, model, alphabet, args.offset_idx
                    ),
                    axis=1,
                )
    # check if output path exists
    if not os.path.exists(dms_output):
        os.makedirs(dms_output)

    dms_df.to_csv(dms_output)