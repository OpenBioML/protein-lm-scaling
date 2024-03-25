# %%
from math import log2
from Bio import SeqIO

import numpy as np
from sklearn.linear_model import LinearRegression
from transformers import AutoTokenizer, AutoModel, EsmModel
import torch
from scipy.sparse.csgraph import minimum_spanning_tree



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
    TODO: either drop similarly to AA freq in seq or smooth?
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
    KLD_dict = {k: v * log2(v/background_freq[k]) for k, v in AA_counts.items() if v != 0} 
    KLD = sum(KLD_dict.values())
    return KLD 


def process_sequence(rec):
    """
    Process a single sequence and return counts and frequencies
    """
    seq = str(rec.seq)
    counts = get_AA_counts(seq, alphabet)
    frequency = get_frquency(seq, alphabet)
    return counts, frequency


def get_background_from_fasta_no_alignment(fasta_file, alphabet, num_seqs):
    """
    iterates over fasta to get the AA frequencies of all seqs.
    """
    fasta_iterator = SeqIO.parse(fasta_file, "fasta")

    # Initialize dictionaries to store counts and frequencies
    total_frequency = {x: 0 for x in alphabet}

    # Iterate over the records in the FASTA file
    for record in fasta_iterator:
        # Get sequence as a string
        seq = str(record.seq)
        
        # Compute counts and frequencies for this sequence
        frequency = get_frquency(seq, alphabet)
        
        # Accumulate counts and frequencies
        for aa in alphabet:
            total_frequency[aa] += frequency[aa]

    # Normalize frequencies by the number of sequences
    num_sequences = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    for aa in alphabet:
        total_frequency[aa] /= num_seqs
    
    return total_frequency

def compute_KLD_fasta(fasta_file, alphabet, background_freq):
    """
    computes the KLD witht the background of the given fasta.
    """

    fasta_iterator = SeqIO.parse(fasta_file, "fasta")

    KLDs = {}

    for rec in fasta_iterator:
        # get ID and seq, pack into dict as id:KLD
        KLDs[rec.id] = compute_kullback_leibler(str(rec.seq), alphabet, background_freq)

    return KLDs

# %%
# intrinsic dimension as suggested by @Amelie-Schreiber
# https://huggingface.co/blog/AmelieSchreiber/intrinsic-dimension-of-proteins

def estimate_persistent_homology_dimension_avg(sequence, model, tokenizer, num_subsets=5, num_iterations=10):
    """
    Estimate the persistent homology dimension of a given protein sequence.
    
    Parameters:
    - sequence: A string representing the protein sequence.
    - model: a model that computes embeddings from the prot seq, tested only with esm atm
    - tokenizer: tokenizer fitting to the model.
    - num_subsets: A positive integer indicating the number of subsets of the embedding vectors to use. Max of 2**n where n=len(sequence). 
    - num_iterations: A positive integer indicating the number of iterations for averaging.
    
    Returns:
    - avg_phd: Average estimated persistent homology dimension.
    """
    
    phd_values = []  # List to store PHD values for each iteration
    
    for _ in range(num_iterations):
        
        # Tokenize the input and convert to tensors
        inputs = tokenizer(sequence, return_tensors='pt')

        # Get the embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[0].numpy()

        # Remove the first and last embeddings (<CLS> and <EOS>)
        embeddings = embeddings[1:-1]

        # Sizes for the subsets to sample
        sizes = np.linspace(2, len(embeddings), num=num_subsets, dtype=int)

        # Prepare data for linear regression
        x = []
        y = []

        for size in sizes:
            # Sample a subset of the embeddings
            subset = np.random.choice(len(embeddings), size, replace=False)
            subset_embeddings = embeddings[subset]

            # Compute the distance matrix
            dist_matrix = np.sqrt(np.sum((subset_embeddings[:, None] - subset_embeddings)**2, axis=-1))

            # Compute the minimum spanning tree
            mst = minimum_spanning_tree(dist_matrix).toarray()

            # Calculate the persistent score E (the maximum edge length in the MST)
            E = np.max(mst)

            # Append to the data for linear regression
            x.append(np.log(size))
            y.append(np.log(E))

        # Reshape for sklearn
        X = np.array(x).reshape(-1, 1)
        Y = np.array(y).reshape(-1, 1)

        # Linear regression
        reg = LinearRegression().fit(X, Y)

        # Estimated Persistent Homology Dimension for this iteration
        phd = 1 / (1 - reg.coef_[0][0])
        
        phd_values.append(phd)
    
    avg_phd = np.mean(phd_values)  # Average over all iterations
    return avg_phd
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

test_fasta_path = "C:/Users/maxsp/Work/prots_test_complexity.fasta"
num_sequences = sum(1 for _ in SeqIO.parse(test_fasta_path, "fasta"))

# %%
# run on test fasta
background = get_background_from_fasta_no_alignment(test_fasta_path, alphabet, num_sequences)
KLD = compute_KLD_fasta(test_fasta_path, alphabet, background)

# %%
# test intrinsic dim stuff
# Load the tokenizer and model
model_path = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = EsmModel.from_pretrained(model_path)

estimate_persistent_homology_dimension_avg(prot_seq, model, tokenizer, num_subsets=2, num_iterations=10)
estimate_persistent_homology_dimension_avg(prot_seq_low_comp, model, tokenizer, num_subsets=2, num_iterations=10)