# %%
from math import log2
from Bio import SeqIO
import pandas as pd
# %%
prot_seq = "ARNDCEQGHILKMFPSTWYVARNDCEQGHILKMFPSTWYV"
prot_seq_halfhalf = "ARNDCEQGHILKMFPSTWYVAAAAAAAAAAAAAAAAAAAA"
prot_seq_low_comp = "MMAAAMMAAAMMAAAMMAAAMMAAAMMAAAMMAAAMMAAA"
prot_seq_homo_rep = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
prot_seq_homo_rep_halflength = "AAAAAAAAAAAAAAAAAAAA"

alphabet = [
    'A',
    'R',
    'N',
    'D',
    'C',
    'E',
    'Q',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V'
]

# %%
# simple metrics
def get_AA_counts(seq, alphabet):
    """computes the counts of AAs in a seq"""
    AA_counts = {x:0 for x in alphabet}
    for AA in alphabet:
        AA_counts[AA] = seq.count(AA)
    return AA_counts

def get_frquency(seq, alphabet):
    """computes the frequency of AAs in seq"""
    AA_counts = get_AA_counts(seq, alphabet)
    return {k: v/len(seq) for k,v in AA_counts.items()}


def compute_entropy(seq, alphabet):
    """
    computes single seq entropy, 
    as zero freq AAs are a possibility, we skip zero frqs in the sumation. 
    I thought that would be better than smoothing with 1 or another constant, but I am not sure. 
    I found a paper where it is also done that way: 
    https://academic.oup.com/mbe/article/40/4/msad084/7111731
    """
    AA_counts = get_frquency(seq, alphabet)
    E_dict = {k: v * log2(v) for k, v in AA_counts.items() if v != 0}
    E = - sum(E_dict.values())
    return E 

def compute_kullback_leibler(seq, alphabet, background_freq=None):
    """
    computesKL-divergence given a background frequency  
    as zero freq AAs are a possibility, we skip zero frqs in the sumation. 
    if AAs in background equal zero, this will also throw an error!
    TODO: either catch if similarly to AA freq in seq or smooth?
    """
    # set background to 0.05 for each if none given
    if background_freq == None:
        background_freq = {'A': 0.05,
            'R': 0.05,
            'N': 0.05,
            'D': 0.05,
            'C': 0.05,
            'E': 0.05,
            'Q': 0.05,
            'G': 0.05,
            'H': 0.05,
            'I': 0.05,
            'L': 0.05,
            'K': 0.05,
            'M': 0.05,
            'F': 0.05,
            'P': 0.05,
            'S': 0.05,
            'T': 0.05,
            'W': 0.05,
            'Y': 0.05,
            'V': 0.05
        }

    AA_counts = get_frquency(seq, alphabet)
    D_kl_dict = {k: v * log2(v/background_freq[k]) for k, v in AA_counts.items() if v != 0} 
    D_kl = - sum(E_dict.values())
    return D_kl 

def compute_KL_div_no_alignment(fasta_file, alphabet):
    """
    Computes Kullback-Leibler Divergence for each seq in fasta.
    Since there is no alignment, it uses the overall average freq of AAs as background:
    Interate over every seq compute frequency and sum up, divide by two each time 
    For very large files, we should include some kind of sampling I guess
    """
    seqs = SeqIO.parse(fasta_file, "fasta")
    background_freq = 
    for rec in seqs:


# %%
test_fasta_path = "C:/Users/maxsp/Work/prots_test_complexity.fasta"

sequences = SeqIO.parse(test_fasta_path, "fasta")

