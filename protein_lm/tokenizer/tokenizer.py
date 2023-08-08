from typing import List

from rust_trie import Trie 


class Tokenizer:
    def __init__(self, tokens: List[str]):
        self.ids_to_tokens = tokens
        self.trie = Trie()
        for token in tokens:
            self.trie.add(token)
        # The Trie will return len(self.ids_to_tokens) when it can't find the token
        # For decoding, we need to add the <unk> token back in, even if it already exists
        self.ids_to_tokens += ["<unk>"]

    def encode(self, sequence: str) -> List[int]:
        return self.trie.tokenize(sequence)

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
        super().__init__(self.ids_to_tokens)


