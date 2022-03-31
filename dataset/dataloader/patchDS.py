import pandas as pd
import numpy as np
import os
import hydra
from torch.utils.data.dataset import Dataset
from utils import generate_patch
from hydra.utils import get_original_cwd
from pathlib import Path


class PatchDataModule(Dataset):
    def __init__(self,
                path_lab,
                path_unlab,
                n,
                deviation_to_shrink_df,
                deviation_for_perfect_hit1,
                deviation_for_perfect_hit2,
                deviation_between_two_points,
                features_metrical,
                features_categorical
                 ):

        self.n = n
        self.path_lab = path_lab
        self.path_unlab = path_unlab
        self.deviation_to_shrink_df = deviation_to_shrink_df
        self.deviation_for_perfect_hit1 = deviation_for_perfect_hit1
        self.deviation_for_perfect_hit2 = deviation_for_perfect_hit2
        self.patch = None
        self.deviation_between_two_points = deviation_between_two_points
        self.features_metrical = features_metrical
        self.features_categorical = features_categorical
        self.all_features = self.features_metrical + self.features_categorical
        self.data_lab = None
        self.data_unlab = None
        self.amount_params = len(self.all_features)

        # generate path for patch
        filestart = "../data/Feb_2022/"
        fileend = "patch_" + str(self.n) + "_" + str(self.all_features)
        file = filestart + fileend + ".npy"
        self.patch_path = Path(os.path.join(get_original_cwd(), file))

        # If patch is already generated, return patch through existing file, else generate patch
        if not self.patch_path.is_file():
            self.generate_patch()
        self.patch = np.load(self.patch_path)

    # Datamodule erstellen - labaledDS orientieren

    def generate_patch(self):
        # prepare data
        self.data_lab = pd.read_csv(self.path_lab)
        self.data_unlab = pd.read_csv(self.path_unlab)
        # prepare features
        self.data_lab = self.data_lab[self.all_features]
        self.data_unlab = self.data_unlab[self.all_features]

        patchgenerator = generate_patch.Patchgenerator(self.data_lab,
                                                       self.data_unlab,
                                                       self.n,
                                                       self.amount_params,
                                                       self.deviation_to_shrink_df,
                                                       self.deviation_for_perfect_hit1,
                                                       self.deviation_for_perfect_hit2,
                                                       self.deviation_between_two_points
                                                       )

        result = patchgenerator.generate_patch()
        np.save(self.patch_path, result)
        return

    def __getitem__(self, item):
        return self.patch[item]

    def __len__(self):
        return len(self.data_lab.index)

@hydra.main(config_path='../../conf', config_name='config')
def test(cfg):
    path_lab = '../../' + cfg.dataset.path_labeled
    path_unlab = '../../' + cfg.dataset.path_unlabeled
    patchDS = PatchDataModule(path_lab=os.path.join(get_original_cwd(), path_lab),
                            path_unlab=os.path.join(get_original_cwd(), path_unlab),
                            n=cfg.patch.parameters.n,
                            deviation_to_shrink_df=cfg.patch.parameters.deviation_to_shrink_df,
                            deviation_for_perfect_hit1=cfg.patch.parameters.deviation_for_perfect_hit1,
                            deviation_for_perfect_hit2=cfg.patch.parameters.deviation_for_perfect_hit2,
                            deviation_between_two_points=cfg.patch.parameters.deviation_between_two_points,
                            features_metrical = cfg.vars.features_metrical,
                            features_categorical = cfg.vars.features_categorical
                              )
    x = patchDS.__getitem__(10)
    c = 0

if __name__ == '__main__':
    test()