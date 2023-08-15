Protein language models scaling laws
==============

The goal of this project is to uncover the best approach to scale large protein language models (ie., learn scaling laws for protein language models) and then publicly release a suite of optimally-trained large protein language models.

## Installing enviroment

```
conda env create -f protein_lm.yml
conda activate protein_lm_env
```

## Installing tokenizer

This will be integrated to the rest of our installation formula, but for now, you need to run the following to build the Rust dependency of the tokenizer:

```
pip install -e protein_lm/tokenizer/rust_trie
```

## Training

There are two options to train the APT Model

- Method 1

An example file is provided in the protein_lm/dataset/uniref folder

```
python3 train.py --train_file_path <path_to_file> --max_sequence_length 20 --checkpoint_dir <path_to_directory> --epochs 10 --steps_per_epoch 2
```
There is an also option to read from a hugging face dataset
In this example, we are reading the [zpn/uniref50](https://huggingface.co/datasets/zpn/uniref50) dataset

```
python3 train.py --hf_dataset_name zpn/uniref50 --max_sequence_length 20 --checkpoint_dir <path_to_directory> --epochs 10 --steps_per_epoch 2
```


- Method 2

```
from train import train
train(train_file_path = '', max_sequence_length = 20, epochs = 10, steps_per_epoch = 2,checkpoint_dir  = '')
```
