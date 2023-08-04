# %%
# huggingface
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from evaluate import load
# others
# import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from datasets import DatasetDict
# http requests
import requests, zipfile, io, os

# %%
# download substitutions, save to disk
path = "data/ProteinGym/"
sub_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip"

if os.path.exists(path + "ProteinGym_substitutions"):
    print("substitution data is already here :)")
else:
    print("download substitution data ...")
    r = requests.get(sub_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

# download indels, save to disk
sub_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_indels.zip"

if os.path.exists(path + "ProteinGym_indels"):
    print("indel data is already here :)")
else:
    print("download indel data ...")
    r = requests.get(sub_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

# %%
# load substitution data and tokenize
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
            all_data.append(df)
    merged_data = pd.concat(all_data, ignore_index=True)
    # Add spaces between each amino acid in the 'mutated_sequences' column
    merged_data['mutated_sequence'] = merged_data['mutated_sequence'].apply(lambda seq: ' '.join(list(seq)))
    # add cls and end tokens
    merged_data['mutated_sequence'] = "[CLS] " + merged_data['mutated_sequence'] + " [EOS]"
    # save csv
    merged_data.to_csv(path + "ProteinGym_substitutions.csv", index=False)
    dataset = load_dataset("csv", data_files=(path + "ProteinGym_substitutions.csv"))
    del merged_data

# %% tokenize
checkpoint = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize(batch):
    return tokenizer(batch["mutated_sequence"], truncation=True, padding='max_length', max_length=760)

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
# # here we could split into validation and test
# dict_test_valid = test_dataset.train_test_split(test_size=0.5, shuffle=True)
# test_dataset = dict_test_valid['test']
# valid_dataset = dict_test_valid['train']
# %%  taken from facebooks pretrained-finetuning notebook here: 
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling.ipynb#scrollTo=fc164b49

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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# %%
trainer.train()