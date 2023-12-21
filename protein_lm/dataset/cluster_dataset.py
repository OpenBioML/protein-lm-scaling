from typing import Callable
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from Bio import SeqIO


class ClusterDataset(Dataset):
    def __init__(
            self, 
            dataset_path: str, 
            cluster_table_path: str,
            size_to_sample_prob: Callable = lambda x: x,
            seed: int = 42,    
        ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.cluster_table_path = cluster_table_path
        self.cluster_to_seqs = {}
        self.cluster_table = pd.read_csv(
            cluster_table_path, dtype={'cluster_name': str, 'cluster_size': int}
        )
        self.cluster_table['sample_prob'] = self.cluster_table['cluster_size'].apply(size_to_sample_prob)
        self.cluster_table['sample_prob'] /= self.cluster_table['sample_prob'].sum()
        self.generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.cluster_table)
    
    def get_cluster_seqs(self, cluster_path: str) -> list:
        if cluster_path not in self.cluster_to_seqs:
            self.cluster_to_seqs[cluster_path] = [
                str(x.seq) for x in SeqIO.parse(cluster_path, 'fasta')
            ]
        return self.cluster_to_seqs[cluster_path]

    def __iter__(self):
        for _ in range(len(self)):
            cluster_name = self.cluster_table.sample(
                n=1, weights='sample_prob', random_state=self.generator
            )[['cluster_name']].values[0][0]
            # Now we map cluster_name to the folder it is in
            if cluster_name == "unk":
                cluster_path = os.path.join(self.dataset_path, "unk", "unk.fasta")
            else:
                cluster_dir = f"{int(cluster_name) // 1000}000"
                cluster_path = os.path.join(self.dataset_path, cluster_dir, f"{cluster_name}.fasta")
            seqs = self.get_cluster_seqs(cluster_path)
            yield seqs[self.generator.integers(len(seqs))]
