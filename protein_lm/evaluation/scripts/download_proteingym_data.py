######################################################################################################
# Script: ProteinGym Download script
# Authors: Maximilian Sprang, Muedi
# Date: 09/2023
# Description: This script downloads and handles the zipped Protein Gym Csvs and preprocesses them 
# to be able to tokenized by EMS/ProtBERT tokenizers.
######################################################################################################
import pandas as pd
import requests, zipfile, io, os
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description='Download ProteinGym Data')
    parser.add_argument("--data_path", default="protein_lm/dataset/ProteinGym/", type=str, help="Path to drop data")
    args = parser.parse_args()
    # relative protgym path
    data_path = args.data_path

    # download substitutions, unzip, save to disk
    dat_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip"

    if os.path.exists(data_path + "ProteinGym_substitutions"):
        print("substitution data is already here :)")
    else:
        print("download substitution data ...")
        r = requests.get(dat_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_path)

    # download indels, unzip, save to disk
    dat_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_indels.zip"

    if os.path.exists(data_path + "ProteinGym_indels"):
        print("indel data is already here :)")
    else:
        print("download indel data ...")
        r = requests.get(dat_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_path)

    # download ref files
    dat_url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/ProteinGym_reference_file_substitutions.csv"
    if os.path.exists(data_path + "ProteinGym_reference_file_substitutions.csv"):
        print("Substitution reference file is already here :)")
    else:
        print("download substitution reference ...")
        r = requests.get(dat_url)
        df = pd.read_csv(io.BytesIO(r.content))
        df.to_csv(data_path + "ProteinGym_reference_file_substitutions.csv", index=False)

    dat_url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/ProteinGym_reference_file_indels.csvv"
    if os.path.exists(data_path + "ProteinGym_reference_file_indels.csv"):
        print("Indel reference file is already here :)")
    else:
        print("download Indel reference ...")
        r = requests.get(dat_url)
        df = pd.read_csv(io.BytesIO(r.content))
        df.to_csv(data_path + "ProteinGym_reference_file_indels.csv", index=False)
    # %%
    # load substitution data, introduce whitespeces and CLS/EOS tokens
    # save complete data as csv, load as HF dataset
    if os.path.exists(data_path + "ProteinGym_substitutions.csv"):
        print("preprocessing was already done, load csv")
        dataset = load_dataset("csv", data_files=(data_path + "ProteinGym_substitutions.csv"))
    else:
        print("preprocess substitutions ...")
        folder_path = "data/ProteinGym/ProteinGym_substitutions"
        all_data = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                experiment = filename[:-4]
        
        
                # add experiment name to track and get base sequence for zero-shot tasks
                df["mutant"] = experiment + "_" + df["mutant"]
                all_data.append(df)
                
        # get dataframe
        merged_data = pd.concat(all_data, ignore_index=True)
        # save the baseseqs
        # Add spaces between each amino acid in the "mutated_sequences" column
        # merged_data["mutated_sequence"] = merged_data["mutated_sequence"].apply(lambda seq: " ".join(list(seq)))
        # add cls and end tokens
        merged_data["mutated_sequence"] = "<cls>" + merged_data["mutated_sequence"] + "<eos>"
        # save csv
        merged_data.to_csv(data_path + "ProteinGym_substitutions.csv", index=False)
        dataset = load_dataset("csv", data_files=(data_path + "ProteinGym_substitutions.csv"))
        del merged_data


if __name__ == '__main__':
    main()