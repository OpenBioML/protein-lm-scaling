# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# huggingface
from transformers import BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset
# others
# import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from datasets import DatasetDict
# http requests
import requests, zipfile, io, os

# %%
# download substitutions, save to disk
path = "data/ProteinGym/"
sub_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_substitutions.zip"

if os.path.exists(path + "ProteinGym_substitutions"):
    print("substitution data is already here :)")
else:
    print("download data ...")
    r = requests.get(sub_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

# download indels, save to disk
sub_url = "https://marks.hms.harvard.edu/proteingym/ProteinGym_indels.zip"

if os.path.exists(path + "ProteinGym_indels"):
    print("indel data is already here :)")
else:
    print("download data ...")
    r = requests.get(sub_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)