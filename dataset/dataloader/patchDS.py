import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import Dataset
from utils import generate_patch
from hydra.utils import get_original_cwd
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class PatchData(Dataset):
    def __init__(self,
                path_lab,
                path_unlab,
                mode,
                fold,
                num_splits,
                n,
                deviation_to_shrink_df,
                deviation_for_perfect_hit1,
                deviation_for_perfect_hit2,
                deviation_between_two_points,
                features_metrical,
                features_categorical,
                 targets=["sand", "silt", "clay"]
                 ):

        self.n = n
        self.fold = fold
        self.mode = mode
        self.num_splits = num_splits
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
        self.targets = targets

        # zu verändern:
        # get_item target --
        # inn cnn --
        # generate_patch() (float32) --
        # get_data_as_np_array() --

        # generate path for patch
        filestart = "dataset/data/Feb_2022/"
        fileend = "patch_" + str(self.n) + "_" + str(self.all_features) + self.mode
        file = filestart + fileend + ".npy"
        self.patch_path = Path(os.path.join(get_original_cwd(), file))
        self.data_lab = self._setup_data(self.fold)
        self.y = self._setup_targets()
        # If patch is already generated, return patch through existing file, else generate patch
        if not self.patch_path.is_file():
            self.generate_patch()
        else:
            self.patch = np.load(self.patch_path)
            self.patch = np.array(self.patch, dtype=np.float32)

    def generate_patch(self):
        # prepare data
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
        self.patch = np.array(result, dtype=np.float32)
        np.save(self.patch_path, result)
        return

    def __getitem__(self, item):
        return self.patch[item], self.y[item]

    def __len__(self):
        return len(self.patch)

    def normalize(self, means=None, std=None):
        if means is None or std is None:
            means = np.mean(self.patch, axis=(0, -1, -2), keepdims=True)
            std = np.std(self.patch, axis=(0, -1, -2), keepdims=True)
        self.patch = (self.patch - means) / std
        return means, std

    def get_data_as_np_array(self):
        return self.patch, self.y

    def _setup_targets(self):
        targets = self.data_lab[self.targets].values.astype(np.float32)
        return targets / np.sum(targets, axis=1, keepdims=True)

    def _split_data(self, df, strategy="location"):
        assert strategy in ["location", "sample"]
        if strategy == "location":
            location_columns = self.features_metrical[:2]
            df.set_index(location_columns, inplace=True)
            locations = df.index.unique()
            location_splits = np.array_split(locations, self.num_splits)
            data_splits = [df.loc[fold].reset_index() for fold in location_splits]
        else:
            data_splits = np.array_split(df, self.num_splits)

        return data_splits

    def _setup_data(self, fold):
        data = pd.read_csv(self.path_lab)
        data = data.sample(frac=1, random_state=42)
        data_splits = self._split_data(data)

        test_fold = fold
        val_fold = (fold + 1) % 10
        train_folds = [i for i in range(10) if i != test_fold and i != val_fold]

        if self.mode == "test":
            data = data_splits[test_fold]
        elif self.mode == "val":
            data = data_splits[val_fold]
        else:
            data = pd.concat([data_splits[i] for i in train_folds])
        return data

class PatchDataModule(pl.LightningDataModule):
    def __init__(self,
                path_lab,
                path_unlab,
                n,
                fold: int = 0,
                mode="val",
                features_metrical=["x", "y"],
                features_categorical=None,
                num_splits=10,
                batch_size=2,
                num_workers=1,
                deviation_to_shrink_df=100,
                deviation_for_perfect_hit1=50,
                deviation_for_perfect_hit2=25,
                deviation_between_two_points=50):
        super(PatchDataModule, self).__init__()

        # save parameters
        self.n = n
        self.mode = mode
        self.path_lab = path_lab
        self.path_unlab = path_unlab
        self.fold = fold
        self.features_metrical = features_metrical
        self.features_categorical = features_categorical
        self.num_splits = num_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.deviation_to_shrink_df = deviation_to_shrink_df
        self.deviation_for_perfect_hit1 = deviation_for_perfect_hit1
        self.deviation_for_perfect_hit2 = deviation_for_perfect_hit2
        self.deviation_between_two_points = deviation_between_two_points


        #internal setup
        self._is_setup = False

    def setup(self, stage=None):
        self.train = PatchData(path_lab=self.path_lab,
                                path_unlab=self.path_unlab,
                                mode="train",
                                fold=self.fold,
                                num_splits=self.num_splits,
                                n = self.n,
                                deviation_to_shrink_df=self.deviation_to_shrink_df,
                                deviation_for_perfect_hit1=self.deviation_for_perfect_hit1,
                                deviation_for_perfect_hit2=self.deviation_for_perfect_hit2,
                                deviation_between_two_points=self.deviation_between_two_points,
                                features_metrical=self.features_metrical,
                                features_categorical=self.features_categorical)

        self.val = PatchData(path_lab=self.path_lab,
                                path_unlab=self.path_unlab,
                                mode=self.mode,
                                fold=self.fold,
                                num_splits=self.num_splits,
                                n = self.n,
                                deviation_to_shrink_df=self.deviation_to_shrink_df,
                                deviation_for_perfect_hit1=self.deviation_for_perfect_hit1,
                                deviation_for_perfect_hit2=self.deviation_for_perfect_hit2,
                                deviation_between_two_points=self.deviation_between_two_points,
                                features_metrical=self.features_metrical,
                                features_categorical=self.features_categorical)  # 86450 85208 86174
        self.test = PatchData(path_lab=self.path_lab,
                                path_unlab=self.path_unlab,
                                mode=self.mode,
                                fold=self.fold,
                                num_splits=self.num_splits,
                                n = self.n,
                                deviation_to_shrink_df=self.deviation_to_shrink_df,
                                deviation_for_perfect_hit1=self.deviation_for_perfect_hit1,
                                deviation_for_perfect_hit2=self.deviation_for_perfect_hit2,
                                deviation_between_two_points=self.deviation_between_two_points,
                                features_metrical=self.features_metrical,
                                features_categorical=self.features_categorical)

        # normalize the data
        self.means, self.std = self.train.normalize()
        self.val.normalize(self.means, self.std)
        self.test.normalize(self.means, self.std)
        self._is_setup = True

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def num_features(self):
        return self.num_features_metrical + self.num_features_categorical

    @property
    def num_features_metrical(self):
        return len(self.features_metrical)

    @property
    def num_features_categorical(self):
        return len(self.features_categorical)

    @property
    def num_data(self, mode="train"):
        if not self._is_setup:
            self.setup()
        if mode == "train":
            return len(self.train)
        elif mode == "val":
            return len(self.val)
        else:
            return len(self.test)

    @property
    def is_setup(self):
        return self._is_setup
"""
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
                            features_categorical = cfg.vars.features_categorical,
                            mode="val"
                              )
    patchDS.setup()
    x = 0

if __name__ == "__main__":
    test()
    """