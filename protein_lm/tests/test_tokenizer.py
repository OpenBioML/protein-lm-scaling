import pytest
import torch
from protein_lm.tokenizer import EsmTokenizer, AptTokenizer

# Test parameters
TOKENIZERS = [EsmTokenizer(), AptTokenizer()]

# 1. Basic Encoding and Decoding
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_basic_encoding_decoding(tokenizer):
    sequence = "LAGERT"
    encoded = tokenizer.encode(sequence)
    decoded = tokenizer.decode(encoded)
    assert decoded == sequence

# 2. Special Tokens Handling
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_special_tokens(tokenizer):
    sequence = "LAGERT"
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    assert encoded[0] == tokenizer.ids_to_tokens.index("<cls>")
    assert encoded[-1] == tokenizer.ids_to_tokens.index("<eos>")
    assert len(encoded) == len(sequence) + 2

# 3. Max Sequence Length Handling
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_max_sequence_length(tokenizer):
    sequence = "LAGERT"
    max_length = 3
    encoded = tokenizer.encode(sequence, max_sequence_length=max_length)
    assert len(encoded) == max_length

# 4. Returning Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_return_tensors(tokenizer):
    sequence = "LAGERT"
    encoded_tensor = tokenizer.encode(sequence, return_tensor=True)
    assert isinstance(encoded_tensor, torch.Tensor)

# 5. Batch Encoding
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    batch_encoded = tokenizer.batch_encode(sequences)
    assert len(batch_encoded) == len(sequences)

# 6. Handling of Unknown Tokens
def test_unknown_tokens():
    tokenizer = AptTokenizer()  # Assuming unknown tokens will be handled as <unk>
    sequence = "XLAGERT"        # "X" is not in the token list
    encoded = tokenizer.encode(sequence)
    assert encoded[0] == tokenizer.ids_to_tokens.index("<unk>")

# Test Batch Encoding with Special Tokens
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_special_tokens(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    batch_encoded = tokenizer.batch_encode(sequences, add_special_tokens=True)
    for encoded in batch_encoded:
        assert encoded[0] == tokenizer.ids_to_tokens.index("<cls>")
        assert encoded[-1] == tokenizer.ids_to_tokens.index("<eos>")

# Test Batch Encoding with Max Sequence Length
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_max_length(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    max_length = 3
    batch_encoded = tokenizer.batch_encode(sequences, max_sequence_length=max_length)
    for encoded in batch_encoded:
        assert len(encoded) == max_length

# Test Batch Encoding Returning Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_return_tensors(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    batch_encoded = tokenizer.batch_encode(sequences, return_tensors=True)
    assert isinstance(batch_encoded, torch.Tensor)

# Test Batch Encoding with Special Tokens, Max Length, and Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_all_combinations(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    max_length = 5
    batch_encoded = tokenizer.batch_encode(
        sequences, 
        add_special_tokens=True, 
        return_tensors=True, 
        max_sequence_length=max_length
    )
    assert isinstance(batch_encoded, torch.Tensor)
    assert batch_encoded.size(1) == max_length

# Test Batch Encoding with Empty List
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_empty_list(tokenizer):
    sequences = []
    batch_encoded = tokenizer.batch_encode(sequences)
    assert batch_encoded == []
