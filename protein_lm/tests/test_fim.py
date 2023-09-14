import pytest
from protein_lm.utils.data_utils import suffix_prefix_middle
from protein_lm.utils.fim_collator import FimDataCollator
from protein_lm.tokenizer.tokenizer import AptTokenizer
import torch


@pytest.mark.parametrize(
    "aa_string, min_span_len, max_span_len, expected_result",
    [
        # Test Case 1: Empty aa_string
        ("", 1, 1, "<cls><eos>"),
        
        # Test Case 2: Single character aa_string
        ("A", 1, 1, "<cls>A<eos>"),
        
        # Test Case 3: aa_string with only two characters
        ("AB", 1, 1, "<cls>AB<eos>"),
        
        # Test Case 4: Standard case
        ("EARRRAGGG", 1, 10, None),  # This could have multiple valid outputs due to randomness
        
        # Test Case 5: max_span_len greater than length of aa_string
        ("EARR", 1, 10, None),  # This could have multiple valid outputs due to randomness
        
        # Test Case 6: min_span_len greater than max_span_len (Should trigger an assertion error)
        ("EARR", 5, 3, AssertionError),
        
        # Test Case 7: min_span_len equal to max_span_len
        ("EARR", 2, 2, None),  # This could have multiple valid outputs due to randomness
        
    ]
)

def test_suffix_prefix_middle(aa_string, min_span_len, max_span_len, expected_result):
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            suffix_prefix_middle(aa_string, min_span_len, max_span_len)
    else:
        result = suffix_prefix_middle(aa_string, min_span_len, max_span_len)
        if expected_result is None:
            assert result != aa_string  # We expect the function to modify the aa_string
        else:
            assert result == expected_result


@pytest.mark.parametrize(
    "aa_string, min_span_len, max_span_len, expected_result",
    [
        # Test Case 1: Empty aa_string
        ("", 1, 1, torch.tensor([0,2])),

        # Test Case 2: Single character aa_string
        ("A", 1, 1, torch.tensor([0,4,2])),

        # Test Case 3: aa_string with only two characters
        ("AB", 1, 1, torch.tensor([0,4,23,2])),

        # Test Case 4: Standard case
        ("EARRRAGGG", 1, 10, None),  # This could have multiple valid outputs due to randomness

        # Test Case 5: max_span_len greater than length of aa_string
        ("EARR", 1, 10, None),  # This could have multiple valid outputs due to randomness
        
        # Test Case 6: min_span_len greater than max_span_len (Should trigger an assertion error)
        ("EARR", 5, 3, AssertionError),

        # Test Case 7: min_span_len equal to max_span_len
        ("EARR", 2, 2, None),  # This could have multiple valid outputs due to randomness

    ]
)

def test_fim_collator(aa_string, min_span_len, max_span_len, expected_result):

    tokenizer=AptTokenizer()
    fim_data_collator = FimDataCollator(
            min_span_len=min_span_len,
            max_span_len=max_span_len,
            fim_freq=0.5,  # You can set this to your desired default value
            mlm=False,  # You can set this to your desired default value
        )

    input_ids = tokenizer.encode(aa_string, add_special_tokens=True)
    input_ids=torch.tensor(input_ids)

        
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(expected_result):
            fim_data_collator.fim_transform(input_ids)
    else:
        # Tokenize aa_string using the tokenize

        result = fim_data_collator.fim_transform(input_ids)
        if expected_result is None:
            assert not torch.equal(result,input_ids)  # We expect the function to modify the aa_string
        else:
            assert torch.equal(result,expected_result)