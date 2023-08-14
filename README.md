Protein language models scaling laws
==============

The goal of this project is to uncover the best approach to scale large protein language models (ie., learn scaling laws for protein language models) and then publicly release a suite of optimally-trained large protein language models.

## Installing enviroment

```

conda env create -f protein_lm.yml
conda activate protein_lm_env
pip install -e .
```

## Installing tokenizer

This will be integrated to the rest of our installation formula, but for now, you need to run the following to build the Rust dependency of the tokenizer:

```
pip install -e protein_lm/tokenizer/rust_trie
```

## Training

There are two options to train the APT Model

At the root project directory (e.g protein_lm_scaling/), use any one of the following methods


### Method 1


An example file is provided in the protein_lm/dataset/uniref folder

```
python3 protein_lm/modeling/scripts/train.py --train_file_path <path_to_file> --sequence_column <name_of_column> --max_sequence_length 1024 --epochs 10 --steps_per_epoch 2
```
There is an also option to read from a hugging face dataset
In this example, we are reading the [zpn/uniref50](https://huggingface.co/datasets/zpn/uniref50) dataset

```
python3 protein_lm/modeling/scripts/train.py --hf_dataset zpn/uniref50 --sequence_column sequence --max_sequence_length 1024 --epochs 10 --steps_per_epoch 2
```


### Method 2

```
from protein_lm.modeling.scripts.train import train
train(train_file_path = '', sequence_column = '', max_sequence_length = 20, epochs = 10, steps_per_epoch = 2)
```
OR

```
from protein_lm.modeling.scripts.train import train
train(hf_dataset = 'zpn/uniref50', sequence_column = 'sequence', max_sequence_length = 20, epochs = 10, steps_per_epoch = 2)
```


## Getting involved
Your involvement is welcome! If you are interested, you can 
- Join the `#protein-lm-scaling` channel on the [OpenBioML discord server](https://discord.com/invite/GgDBFP8ZEt).
- Check out our [contributing guide](docs/CONTRIBUTING.md) if you are interested in contributing to this repository.
