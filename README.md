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

### Toy using local dataset

We recommend using a toy tiny dataset for testing and debugging new changes that do not rely on having a large datset. Such a small dataset is provided in the `protein_lm/dataset/uniref` folder and an example toy training config yaml that uses this dataset is provided in `protein_lm/configs/train/toy_localcsv.yaml`. To use this config, at the root project directory (e.g., `protein_lm_scaling/`), run

```
python protein_lm/modeling/scripts/train.py --config-file protein_lm/configs/train/toy_localcsv.yaml
```

This config is actually the default, so the above is equivalent to

```
python protein_lm/modeling/scripts/train.py
```

### Toy using a HuggingFace dataset

For testing with a HuggingFace dataset, we have an example config yaml in `protein_lm/configs/train/toy_hf.yaml`. Note that training with this config is a little more involved than the above `protein_lm/configs/train/toy_localcsv.yaml`:

* When first run, the script will download the [processed uniref50 dataset](https://huggingface.co/datasets/zpn/uniref50), which could take some time.
* This config will log the loss values and other metrics to Weights and Biases. This will require you to create a wandb account.

You can run with this config by:

```
python protein_lm/modeling/scripts/train.py --config-file protein_lm/configs/train/toy_hf.yaml
```

### Running on multiple gpus

We can run on a single node with multiple gpus by

```
torchrun --standalone --nnodes=1 --nproc-per-node <num_gpus> protein_lm/modeling/scripts/train.py --config-file <config_file>
```

For example, to run on a single node with 3 gpus with the provided `protein_lm/configs/train/toy_hf.yaml` config file, we can run with

```
torchrun --standalone --nnodes=1 --nproc-per-node 3 protein_lm/modeling/scripts/train.py --config-file protein_lm/configs/train/toy_hf.yaml
```

## Getting involved
Your involvement is welcome! If you are interested, you can 
- Join the `#protein-lm-scaling` channel on the [OpenBioML discord server](https://discord.com/invite/GgDBFP8ZEt).
- Check out our [contributing guide](docs/CONTRIBUTING.md) if you are interested in contributing to this repository.
