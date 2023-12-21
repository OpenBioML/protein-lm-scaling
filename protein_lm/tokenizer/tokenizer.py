import torch
from typing import List, Union, Optional

from rust_trie import Trie 


class Tokenizer:
    def __init__(self, tokens: List[str], unk_token_id: Optional[int] = None):
        self.ids_to_tokens = tokens
        self.trie = Trie(unk_token_id)
        for token in tokens:
            self.trie.add(token)
        # If unk_token_id is not provided, add <unk> to the end of the tokens list
        if unk_token_id is None:
            self.ids_to_tokens += ["<unk>"]
        self.pad_token_id = self.ids_to_tokens.index("<pad>")
        self.mask_token_id = self.ids_to_tokens.index("<mask>")


    def __call__(self, sequences: Union[str, List], *args, **kwargs):
        if isinstance(sequences, str):
            return self.encode(sequences, *args, **kwargs)
        else:
            return self.batch_encode(sequences, *args, **kwargs)
    
    def encode(
        self, 
        sequence: str, 
        add_special_tokens: bool = False,
        return_tensor: bool = False,
        max_sequence_length: Optional[int] = None,
    ) -> List[int]:
        if max_sequence_length is not None:
            if add_special_tokens:
                max_sequence_length -= 2
            sequence = sequence[:max_sequence_length]
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
        max_sequence_length: Optional[int] = None,
    ) -> List[List[int]]:
        output = []
        if max_sequence_length is None and return_tensors:
            max_sequence_length = max([len(sequence) for sequence in sequences])
            if add_special_tokens:
                max_sequence_length += 2
        if max_sequence_length is not None:
            sequences = [
                sequence[:(max_sequence_length - 2) if add_special_tokens else max_sequence_length] 
                for sequence in sequences
            ]
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
        tokens = [
            "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", 
            "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", 
            "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", 
            "Z", "O", ".", "-", "<null_1>", "<mask>"
        ]
        super().__init__(tokens, unk_token_id=3)



class AptTokenizer(Tokenizer):
    def __init__(self):
        # For our own tokenizers, we don't need to explicitly add the <unk> token
        # because it gets added as the last token in the tokens list
        # I've also removed X so that it gets translated to <unk>
        tokens = [
            "<cls>", "<pad>", "<eos>", "L", "A", "G", "V", 
            "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", 
            "F", "Y", "M", "H", "W", "C", "B", "U", "Z", "O", 
            "<mask>"
        ]
        super().__init__(tokens)