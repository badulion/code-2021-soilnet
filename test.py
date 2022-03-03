import pandas as pd
import dask.dataframe as dd
import numpy as np
import os
from tqdm import tqdm
import sys

path = "dataset/data/Feb_2022/fr_lab.csv"
df = pd.read_csv(path)
print(df.dtypes)

#ddf = df.reset_index().compute()
print(pd.unique(df["Bezirk"]))
print(pd.unique(df["BK1000"]))
