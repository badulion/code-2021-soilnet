import pandas as pd
import numpy as np
import hydra
from hydra.utils import get_original_cwd
import os


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg):
    df_path = os.path.join(get_original_cwd(), cfg.dataset.path_labeled)
    df = pd.read_csv(df_path)
    df = df.rename(columns={"S_GES": "sand", "U_GES": "silt", "T": "clay"})
    df["depth"] = (df["PUT"]+df["POT"])/2
    df.to_csv(df_path, index=False)


if __name__ == '__main__':
    my_app()
