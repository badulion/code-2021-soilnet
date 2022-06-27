import hydra
from hydra.utils import get_original_cwd
import os
import pandas as pd

@hydra.main(config_path='conf', config_name='config')
def start(cfg):
    path_labeled = os.path.join(get_original_cwd(), cfg.dataset.path_labeled)
    df_labeled = pd.read_csv(path_labeled)
    df_labeled.drop(df_labeled[(df_labeled.POT > 4) | (df_labeled.PUT < 0)].index, inplace=True)
    df_labeled.drop(['depth'], axis=1, inplace=True)
    path = 'dataset/data/Feb_2022/fr_lab_neu_mit_0_bis_4.csv'
    df_labeled.to_csv(os.path.join(get_original_cwd(), path))



if __name__ == '__main__':
    start()
