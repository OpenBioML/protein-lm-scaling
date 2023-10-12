# %%
######################################################################################################
# Script: ProteinGym Supervised Eval Script
# Authors: Maximilian Sprang, Muedi
# Date: 09/2023
# Description: This script uses HF's evaluate library to test supervised perfromance of a given model 
# on ProteinGym data.
# ATM only substitution data is implemented for the finetunning but both are preprocessed and the
# complete datasets saved as CSV.
######################################################################################################
import sys, os
sys.path.append(os.getcwd()) #needed to run script from base dir.
# Otherwise prot_lm throws module not found exception

# huggingface
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
from datasets import load_dataset
# ours 
from protein_lm.tokenizer import EsmTokenizer, AptTokenizer
# others
from tqdm import tqdm
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Supervised Training Script")
    parser.add_argument("--data_path", default="protein_lm/dataset/ProteinGym/", type=str, help="Path to ProteinGym data")
    parser.add_argument("--checkpoint", default="facebook/esm2_t33_650M_UR50D", type=str, help="Checkpoint, of online model, or path to local checkpoints")
    
    args = parser.parse_args()
    
    checkpoint = args.checkpoint
    data_path = args.data_path
    dataset = load_dataset("csv", data_files=(data_path + "ProteinGym_substitutions.csv"))
    
    # load model for seq classification
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    model.config.pad_token_id = 2     # needed for apt as long as tokenizer is not API compatible :)

    model_name = checkpoint.split("/")[-1]
    batch_size = 8

    tokenizer = AptTokenizer()

    def tokenize(batch):
        tokens = tokenizer(batch["mutated_sequence"], return_tensors=True, max_sequence_length=760)
        return {"input_ids": tokens}

    token_data = dataset.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # rename and remove stuff to fit into dataloader seemlessly
    # removes info we don"t use in the network, as we only use tokens and binned scores
    token_data = token_data.remove_columns(["DMS_score", "mutant", "mutated_sequence"])
    # binned scores are renamed to "labels"
    token_data = token_data.rename_column("DMS_score_bin", "labels")

    # Split the train dataset into train, valid, and test subsets
    dict_train_test = token_data["train"].train_test_split(test_size=0.4, shuffle=True)
    train_dataset = dict_train_test["train"]
    test_dataset = dict_train_test["test"]

    # subset for testruns:
    # train_dataset = train_dataset.select([x for x in range(200)])
    # test_dataset = test_dataset.select([x for x in range(100)])

    # # here we could split into validation and test if needed
    # dict_test_valid = test_dataset.train_test_split(test_size=0.5, shuffle=True)
    # test_dataset = dict_test_valid["test"]
    # valid_dataset = dict_test_valid["train"]

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
    # and save this to the disk in the model-ceckpoint* folder
    trainer.train()

if __name__ == '__main__':
    main()