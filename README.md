Protein language models scaling laws
==============

The goal of this project is to uncover the best approach to scale large protein language models (ie., learn scaling laws for protein language models) and then publicly release a suite of optimally-trained large protein language models.

## Installing tokenizer

This will be integrated to the rest of our installation formula, but for now, you need to run the following to build the Rust dependency of the tokenizer:

```
pip install -e protein_lm/tokenizer/rust_trie
```
