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

# 5. Encoding with Special Tokens
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encoding_special_tokens(tokenizer):
    sequence = "LAGERT"
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    assert encoded[0] == tokenizer.ids_to_tokens.index("<cls>")
    assert encoded[-1] == tokenizer.ids_to_tokens.index("<eos>")
    assert len(encoded) == len(sequence) + 2

# 6. Encoding with Max Sequence Length
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encoding_max_length(tokenizer):
    sequence = "LAGERT"
    max_length = 3
    encoded = tokenizer.encode(sequence, max_sequence_length=max_length)
    assert len(encoded) == max_length

# 7. Encoding Returning Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encoding_return_tensors(tokenizer):
    sequence = "LAGERT"
    encoded_tensor = tokenizer.encode(sequence, return_tensor=True)
    assert isinstance(encoded_tensor, torch.Tensor)

# 8. Encoding with Special Tokens and Max Length
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encoding_special_tokens_max_length(tokenizer):
    sequence = "LAGERT"
    max_length = 3
    encoded = tokenizer.encode(sequence, add_special_tokens=True, max_sequence_length=max_length)
    assert len(encoded) == max_length

# 9. Encoding with Special Tokens and Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encoding_special_tokens_tensors(tokenizer):
    sequence = "LAGERT"
    encoded_tensor = tokenizer.encode(sequence, add_special_tokens=True, return_tensor=True)
    assert isinstance(encoded_tensor, torch.Tensor)
    assert encoded_tensor[0] == tokenizer.ids_to_tokens.index("<cls>")
    assert encoded_tensor[-1] == tokenizer.ids_to_tokens.index("<eos>")

# 10. Encoding with Max Length and Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encoding_max_length_tensors(tokenizer):
    sequence = "LAGERT"
    max_length = 3
    encoded_tensor = tokenizer.encode(sequence, max_sequence_length=max_length, return_tensor=True)
    assert isinstance(encoded_tensor, torch.Tensor)
    assert len(encoded_tensor) == max_length

# 11. Encoding with Special Tokens, Max Length, and Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encoding_all_combinations(tokenizer):
    sequence = "LAGERT"
    max_length = 3
    encoded_tensor = tokenizer.encode(sequence, add_special_tokens=True, max_sequence_length=max_length, return_tensor=True)
    assert isinstance(encoded_tensor, torch.Tensor)
    assert len(encoded_tensor) == max_length


# 12. Batch Encoding
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    batch_encoded = tokenizer.batch_encode(sequences)
    assert len(batch_encoded) == len(sequences)

# 13. Handling of Unknown Tokens
def test_unknown_tokens():
    tokenizer = AptTokenizer()  # Assuming unknown tokens will be handled as <unk>
    sequence = "XLAGERT"        # "X" is not in the token list
    encoded = tokenizer.encode(sequence)
    assert encoded[0] == tokenizer.ids_to_tokens.index("<unk>")

# 14.Test Batch Encoding with Special Tokens
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_special_tokens(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    batch_encoded = tokenizer.batch_encode(sequences, add_special_tokens=True)
    for encoded in batch_encoded:
        assert encoded[0] == tokenizer.ids_to_tokens.index("<cls>")
        assert encoded[-1] == tokenizer.ids_to_tokens.index("<eos>")

# 15.Test Batch Encoding with Max Sequence Length
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_max_length(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    max_length = 3
    batch_encoded = tokenizer.batch_encode(sequences, max_sequence_length=max_length)
    for encoded in batch_encoded:
        assert len(encoded) == max_length

# 16.Test Batch Encoding Returning Tensors
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_return_tensors(tokenizer):
    sequences = ["LAGERT", "SERPK"]
    batch_encoded = tokenizer.batch_encode(sequences, return_tensors=True)
    assert isinstance(batch_encoded, torch.Tensor)

# 17.Test Batch Encoding with Special Tokens, Max Length, and Tensors
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

# 18.Test Batch Encoding with Empty List
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_batch_encoding_empty_list(tokenizer):
    sequences = []
    batch_encoded = tokenizer.batch_encode(sequences)
    assert batch_encoded == []
