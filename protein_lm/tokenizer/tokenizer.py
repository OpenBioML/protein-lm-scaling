import torch
from typing import List

from rust_trie import Trie 


class Tokenizer:
    def __init__(self, tokens: List[str], unk_token_id: int = None):
        self.ids_to_tokens = tokens
        self.trie = Trie(unk_token_id)
        for token in tokens:
            self.trie.add(token)
        # The Trie will return len(self.ids_to_tokens) when it can't find the token
        # For decoding, we need to add the <unk> token back in, even if it already exists
        if unk_token_id is None:
            self.ids_to_tokens += ["<unk>"]
        self.pad_token_id = self.ids_to_tokens.index("<pad>")
        self.mask_token_id = self.ids_to_tokens.index("<mask>")

    def encode(
        self, 
        sequence: str, 
        add_special_tokens: bool = False,
        return_tensor: bool = False,
    ) -> List[int]:
        if add_special_tokens:
            sequence = "<cls>" + sequence + "<eos>"
        output = self.trie.tokenize(sequence)
        if return_tensor:
            output = torch.tensor(output, dtype=torch.long)
        return output

    def batch_encode(
        self,
        sequences: List[str],
        add_special_tokens: bool = False,
        return_tensors: bool = False,
        max_sequence_length: int = None,
    ) -> List[List[int]]:
        output = []
        if max_sequence_length is not None:
            sequences = [sequence[:max_sequence_length] for sequence in sequences]
        for sequence in sequences:
            output.append(self.encode(sequence, add_special_tokens, return_tensors))
        if return_tensors:
            tensor_out = torch.full((len(output), max_sequence_length), self.pad_token_id)
            for i, sequence in enumerate(output):
                tensor_out[i, :len(sequence)] = sequence
            output = tensor_out
        return output

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.ids_to_tokens[idx] for idx in tokens])


class EsmTokenizer(Tokenizer):
    def __init__(self):
        self.ids_to_tokens = [
            "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", 
            "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", 
            "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", 
            "Z", "O", ".", "-", "<null_1>", "<mask>"
        ]
        super().__init__(self.ids_to_tokens, unk_token_id=3)



class AptTokenizer(Tokenizer):
    def __init__(self, tokens: List[str]):
        # For our own tokenizers, we don't need to explicitly add the <unk> token
        # because it gets added as the last token in the tokens list
        # I've also removed X so that it gets translated to <unk>
        self.ids_to_tokens = [
            "<cls>", "<pad>", "<eos>", "L", "A", "G", "V", 
            "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", 
            "F", "Y", "M", "H", "W", "C", "B", "U", "Z", "O", 
            "<mask>"
        ]
        super().__init__(tokens)
