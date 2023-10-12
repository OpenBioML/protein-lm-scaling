 # %%
######################################################################################################
# Script: ProteinGym Supervised Eval Script
# Authors: Maximilian Sprang, Muedi
# Date: 09/2023
# Description: Zero-shot eval script for MLM models like ESM: 
# https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py
######################################################################################################
import sys, os
sys.path.append(os.getcwd()) #needed to run script from base dir.
# Otherwise prot_lm throws module not found exception
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from Bio import SeqIO
import itertools
from typing import List, Tuple
import argparse
# ours
from protein_lm.tokenizer import AptTokenizer
# esm 
from esm import pretrained, Alphabet, FastaBatchedDataset, MSATransformer


# def remove_insertions(sequence: str) -> str:
#     """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
#     # This is an efficient way to delete lowercase characters and insertion characters from a string
#     deletekeys = dict.fromkeys(string.ascii_lowercase)
#     deletekeys["."] = None
#     deletekeys["*"] = None

#     translation = str.maketrans(deletekeys)
#     return sequence.translate(translation)


# def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
#     """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
    
#     The input file must be in a3m format (although we use the SeqIO fasta parser)
#     for remove_insertions to work properly."""

#     msa = [
#         (record.description, remove_insertions(str(record.seq)))
#         for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
#     ]
#     return msa


def label_row(rows, sequence, token_probs, alphabet, offset_idx):
    rows = rows.split(":")
    score = 0
    for row in rows:
        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score_obj = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        score += score_obj.item()
    return score / len(rows)


def compute_pppl(mutated_sequence, model, alphabet):
    """
    The original methods changes the given base_sequence to the mutated one, we"ll just read it from the df.
    We compute the pseudo-Perplexity of the complete mutated sequence. 
    The code to achieve this has not been changed from esm's repo
    """
    # wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    # assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # # modify the sequence
    # sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", mutated_sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute token probabilities at each position
    log_probs = []
    for i in range(1, len(mutated_sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(mutated_sequence[i])].item())  # vocab size
    return sum(log_probs)



def main():
    parser = argparse.ArgumentParser(description="Supervised Training Script")
    parser.add_argument("--checkpoint", default="facebook/esm2_t33_650M_UR50D", type=str, help="Checkpoint, path to local checkpoints")
    parser.add_argument("--data_path", default="protein_lm/dataset/ProteinGym/", type=str, help="Path to ProteinGym data")
    parser.add_argument("--outdir", default="protein_lm/evaluation/output/likelihood-autoreg/", type=str, help="Directory for output files")
    parser.add_argument("--scoring_strategy", default="masked-marginals", choices=["masked-marginals", "pseudo-ppl", "wt-marginals"], type=str, help="Scoring strategies for MLMs")
    parser.add_argument("--nogpu", default=False, type=bool, help="Set true to run model on CPU")
    args = parser.parse_args()
    
    # assign vars
    checkpoint = args.checkpoint
    data_path = args.data_path
    outdir = args.outdir
    nogpu = args.nogpu
    scoring_strategy = args.scoring_strategy
    
    # fixed vars (can be added as args if needed later
    mutation_col = 0 # column that holds info on mutations
    offset_idx = 1 # offset index, default was zero, but in our case it needs to be one

    # get experiments and base seqs
    ref_df = pd.read_csv(data_path + "ProteinGym_reference_file_substitutions.csv")
    dms_ids = ref_df.DMS_id
    dms_file = ref_df.DMS_filename
    dms_ref_seqs = ref_df.target_seq

    # relative output path 
    outdir = "protein_lm/evaluation/output/{}/".format(scoring_strategy)
    # check if output path exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # inference for given model
    # set checkpoint to be mnodel location for now
    model_name = checkpoint.split("/")[-1]
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    for experiment, file_name, sequence in zip(dms_ids, dms_file, dms_ref_seqs):

        # Load the deep mutational scan
        DMS_data_path = data_path + "ProteinGym_substitutions/" + file_name
        DMS_data = pd.read_csv(DMS_data_path)
        DMS_output =  "scores_{}".format(file_name)


        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):
            # as far as I know we do not plan on using this? I kept it around for now.
            print("MSATransformer is currently not supported :)")
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
            # DMS_data[model_name+"_score"] = DMS_data.apply(
            #     lambda row: label_row(
            #         row[mutation_col], sequence, token_probs, alphabet, offset_idx
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
                DMS_data[model_name+"_score"] = DMS_data.apply(
                    lambda row: label_row(
                        row[mutation_col],
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
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
                DMS_data[model_name+"_score"] = DMS_data.apply(
                    lambda row: label_row(
                        row[mutation_col],
                        sequence,
                        token_probs,
                        alphabet,
                        offset_idx,
                    ),
                    axis=1,
                )
            elif scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                DMS_data[model_name+"_score"] = DMS_data.progress_apply(
                    lambda row: compute_pppl(
                        #row[mutation_col],
                        # sequence,
                        row["mutated_sequence"],
                        model,
                        alphabet
                        #offset_idx
                    ),
                    axis=1,
                )
        # save experiment
        DMS_data.to_csv(outdir + DMS_output, index=None)
        spearman, _ = spearmanr(DMS_data["{}_score".format(model_name)], DMS_data["DMS_score"])
        print("Performance of {} on experiment {}: {}".format(model_name, experiment, spearman))

if __name__ == "__main__":
    main()