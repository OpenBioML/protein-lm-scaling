Protein language models scaling laws
==============

The goal of this project is to uncover the best approach to scale large protein language models (ie., learn scaling laws for protein language models) and then publicly release a suite of optimally-trained large protein language models.

## Installing enviroment

If you want to run on CPU:
```
conda env create -f protein_lm.yml
conda activate protein_lm_env
pip install -e .
```

If you plan to use cuda, use the dedicated .yaml file:
```
conda env create -f protein_lm_cuda.yml
conda activate protein_lm_env
pip install -e .
```


## Installing tokenizer

This will be integrated to the rest of our installation formula, but for now, you need to run the following to build the Rust dependency of the tokenizer:

```
pip install -e protein_lm/tokenizer/rust_trie
```

## Training

An example data file is provided in the `protein_lm/dataset/uniref` folder and an example toy training config yaml that uses this dataset is provided: `protein_lm/configs/train/toy_localcsv.yaml`. To use this config, at the root project directory (e.g., `protein_lm_scaling/`), run

```
python protein_lm/modeling/scripts/train.py --config-file protein_lm/configs/train/toy_localcsv.yaml
```

This config is actually the default, so the above is equivalent to

```
python protein_lm/modeling/scripts/train.py
```

An example config yaml of using a dataset from huggingface is `protein_lm/configs/train/toy_hf.yaml`, which you can run with

```
python protein_lm/modeling/scripts/train.py --config-file protein_lm/configs/train/toy_hf.yaml
```


## Getting involved
Your involvement is welcome! If you are interested, you can 
- Join the `#protein-lm-scaling` channel on the [OpenBioML discord server](https://discord.com/invite/GgDBFP8ZEt).
- Check out our [contributing guide](docs/CONTRIBUTING.md) if you are interested in contributing to this repository.
