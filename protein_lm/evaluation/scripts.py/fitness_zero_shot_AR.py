 # %%
######################################################################################################
# Script: ProteinGym Supervised Eval Script
# Authors: Maximilian Sprang, Muedi
# Date: 09/2023
# Description: zero shot for autoregressive models, as fopund in RITA
# https://github.com/lightonai/RITA/blob/master/compute_fitness.py
######################################################################################################
import sys, os
sys.path.append(os.getcwd()) #needed to run script from base dir.
# Otherwise prot_lm throws module not found exception
from transformers import AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import argparse
# ours
from protein_lm.tokenizer import AptTokenizer


def calc_fitness(model, prots, tokenizer, device="cuda:0", model_context_len=1023):
    # calculates the fitness
    loss_list = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad():
        for prot in tqdm(prots):
            loss_val = 0
            
            sequence_chunks=[]
            if len(prot) < model_context_len:
                sequence_chunks = [prot]
            else:
                len_target_seq = len(prot)
                num_windows = 1 + int( len_target_seq / model_context_len)
                start=0
                for window_index in range(1, num_windows+1):
                    sequence_chunks.append(prot[start:start+model_context_len])
                    start += model_context_len
            
            for chunk in sequence_chunks:
                for p in [chunk, chunk[::-1]]:
                    ids = torch.tensor([tokenizer.encode(p)]).to(device)
                    input_ids = ids[:, :-1]
                    targets   = ids[:, 1:]
                    
                    logits=model(input_ids).logits
                    loss = loss_fn(target=targets.view(-1), input=logits.view(-1,logits.size(-1)))
                    loss_val += -loss.item()
                
            loss_list += [loss_val]
    return np.array(loss_list)

def main():
    parser = argparse.ArgumentParser(description="Supervised Training Script")
    parser.add_argument("--checkpoint", default="checkpoints/toy", type=str, help="Checkpoint, path to local checkpoints")
    parser.add_argument("--data_path", default="protein_lm/dataset/ProteinGym/", type=str, help="Path to ProteinGym data")
    parser.add_argument("--outdir", default="protein_lm/evaluation/output/likelihood-autoreg/", type=str, help="Directory for output files")
    parser.add_argument("--nogpu", default=False, type=bool, help="Set true to run model on CPU")
    args = parser.parse_args()
    
    checkpoint = args.checkpoint
    data_path = args.data_path
    outdir = args.outdir
    nogpu = args.nogpu
    # check if output path exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    model.eval()
    tokenizer = AptTokenizer()

    # get experiments and base seqs
    ref_df = pd.read_csv(data_path + "ProteinGym_reference_file_substitutions.csv")
    dms_ids = ref_df.DMS_id
    dms_file = ref_df.DMS_filename
    dms_ref_seqs = ref_df.target_seq
    for experiment, file_name, sequence in zip(dms_ids, dms_file, dms_ref_seqs):

        # Load the deep mutational scan
        DMS_data_path = data_path + "ProteinGym_substitutions/" + file_name
        DMS_data = pd.read_csv(DMS_data_path)
        DMS_output =  "scores_{}".format(file_name)

        # compute scores
        model_scores = calc_fitness(model=model, prots=np.array(DMS_data["mutated_sequence"]), tokenizer=tokenizer)

        DMS_data["APT_score"] = model_scores
        DMS_data.to_csv(outdir + DMS_output, index=False)

        spearman, _ = spearmanr(DMS_data["APT_score"], DMS_data["DMS_score"])
        print("Performance of APT on experiment {}: {}".format(experiment, spearman))

if __name__ == "__main__":
    main()