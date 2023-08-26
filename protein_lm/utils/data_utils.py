import random


def suffix_prefix_middle(
    aa_string: str, 
    min_span_len: int = 1, 
    max_span_len: int = 10,
    masked_span_token: str = "<middle_span>",
    start_token: str = "<cls>",
    end_token: str = "<eos>",
    end_of_span_token: str = "<end_of_span>",
) -> str:
    """
    Given an amino acid string, returns a string with a span of amino acids removed from the middle,
    moved to the end, and special tokens added.
    Thus, if we have EARRRAGGG, we could get <cls>EARRR<middle_span>GG<eos>AG<end_of_span>
    """
    assert min_span_len <= max_span_len, "min_span_len cannot be larger than max_span_len"
    if len(aa_string) < 3:
        return f"{start_token}{aa_string}{end_token}"
    max_span_len = min(max_span_len, len(aa_string))
    if min_span_len == max_span_len:
        span_len = min_span_len
    else:
        span_len = random.randint(min_span_len, max_span_len)
    span_start = random.randint(0, len(aa_string) - span_len)
    span_end = span_start + span_len
    new_aa_string = f"{start_token}{aa_string[:span_start]}{masked_span_token}{aa_string[span_end:]}{end_token}{aa_string[span_start:span_end]}{end_of_span_token}"
    return new_aa_string
