import torch
import pandas as pd
import numpy as np
import os
import hydra
import category_encoders as ce
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pytorch_lightning as pl


class LabeledSoilData(Dataset):
    """A dataset containing soil grid data for Lower, Middle and Upper Franconia."""

    def __init__(self,
                 path,
                 mode: str = "train",
                 fold: int = 0,
                 features_metrical=["x", "y"],
                 features_categorical=None,
                 levels_categorical=None,
                 encoding_categorical="onehot",
                 targets=["sand", "silt", "clay"],
                 num_splits=10,
                 bezirk_column="Bezirk",
                 bezirk=None,
                 ):

        self.path = path
        self.fold = fold

        assert mode in ["train", "val", "test"]
        self.mode = mode

        # save parameters
        self.features_metrical = features_metrical
        self.features_categorical = features_categorical
        self.levels_categorical = levels_categorical
        self.encoding_categorical = encoding_categorical
        self.targets = targets
        self.num_splits = num_splits
        self.bezirk = bezirk
        self.bezirk_column = bezirk_column

        # prepare data
        self.data = self._setup_data(fold)
        self.y = self._setup_targets()
        self.x_categorical = self._setup_cat_features()
        self.x_metric = self._setup_metric_features()
        self.x = np.hstack([self.x_metric, self.x_categorical])

        # depth column
        if "depth" in self.features_metrical:
            self.depths = self.data[["depth"]]
        else:
            self.depths = None

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
        data = pd.read_csv(self.path)
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

        if self.bezirk is not None:
            return data.loc[[b in self.bezirk for b in data[self.bezirk_column]], :]
        else:
            return data

    def _setup_targets(self):
        targets = self.data[self.targets].values.astype(np.float32)
        return targets / np.sum(targets, axis=1, keepdims=True)

    def _setup_cat_features(self):
        if self.features_categorical is None:
            return np.empty((len(self.data), 0), dtype=np.float32)

        x_categorical = self.data[self.features_categorical].values.astype(str)

        cat_encoder = OneHotEncoder(categories=[self.levels_categorical[feature] for feature in self.features_categorical],
                                    sparse=False,
                                    dtype=np.float32)

        x_categorical = cat_encoder.fit_transform(x_categorical)
        return x_categorical

    def _setup_metric_features(self):
        x_metric = self.data[self.features_metrical].values.astype(np.float32)
        return x_metric

    def normalize(self, scaler=None, fit=False):
        if scaler == None:
            scaler = self._create_scaler()

        func = scaler.fit_transform if fit else scaler.transform

        self.x_metric = func(self.x_metric)

        self.x = np.hstack([self.x_metric, self.x_categorical])

    def _create_scaler(self):
        scaler = StandardScaler()
        return scaler

    def __getitem__(self, i: int) -> tuple:
        x = self.x[i, :]
        y = self.y[i, :]

        return x, y

    def __len__(self) -> int:
        return len(self.x)

    def inverse_normalize(self, scaler):
        self.x = scaler.inverse_transform(self.x)

    def get_data_as_np_array(self):
        return self.x, self.y

    def get_permuted_feature(self, feature_id):
        assert feature_id < len(self.features_metrical) + len(self.features_categorical)
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)

        if feature_id < len(self.features_metrical):
            x_metric = self.x_metric.copy()
            x_metric[:, feature_id] = x_metric[indices, feature_id]
            x_shuffled = np.hstack([x_metric, self.x_categorical])
            return x_shuffled, self.y
        else:
            feature_id -= len(self.features_metrical)
            x_categorical = self.x_categorical.copy()
            x_categorical[:, feature_id] = x_categorical[indices, feature_id]
            x_shuffled = np.hstack([self.x_metric, x_categorical])
            return x_shuffled, self.y


class LabeledDataModule(pl.LightningDataModule):
    def __init__(self,
                 path,
                 fold: int = 0,
                 mode="val",
                 features_metrical=["x", "y"],
                 features_categorical=None,
                 levels_categorical=None,
                 encoding_categorical="onehot",
                 targets=["sand", "silt", "clay"],
                 num_splits=10,
                 batch_size=16,
                 num_workers=1):
        super(LabeledDataModule, self).__init__()

        # save parameters
        self.mode = mode
        self.path = path
        self.fold = fold
        self.features_metrical = features_metrical
        self.features_categorical = features_categorical
        self.levels_categorical = levels_categorical
        self.encoding_categorical = encoding_categorical
        self.targets = targets
        self.num_splits = num_splits
        self.batch_size = batch_size
        self.num_workers = num_workers

        #internal setup
        self._is_setup = False

    def setup(self, stage=None):
        self.train = LabeledSoilData(path=self.path,
                                     mode="train",
                                     fold=self.fold,
                                     features_metrical=self.features_metrical,
                                     features_categorical=self.features_categorical,
                                     levels_categorical=self.levels_categorical,
                                     encoding_categorical=self.encoding_categorical,
                                     targets=self.targets,
                                     num_splits=self.num_splits,
                                     bezirk=None)

        self.val = LabeledSoilData(path=self.path,
                                   mode=self.mode,
                                   fold=self.fold,
                                   features_metrical=self.features_metrical,
                                   features_categorical=self.features_categorical,
                                   levels_categorical=self.levels_categorical,
                                   encoding_categorical=self.encoding_categorical,
                                   targets=self.targets,
                                   num_splits=self.num_splits,
                                   bezirk=None)  # 86450 85208 86174
        self.test = LabeledSoilData(path=self.path,
                                    mode=self.mode,
                                    fold=self.fold,
                                    features_metrical=self.features_metrical,
                                    features_categorical=self.features_categorical,
                                    levels_categorical=self.levels_categorical,
                                    encoding_categorical=self.encoding_categorical,
                                    targets=self.targets,
                                    num_splits=self.num_splits,
                                    bezirk=None)

        # normalize the data
        self.scaler = StandardScaler()
        self.train.normalize(self.scaler, fit=True)
        self.val.normalize(self.scaler, fit=False)
        self.test.normalize(self.scaler, fit=False)

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
    def is_setup(self):
        return self._is_setup

    @property
    def num_features(self):
        return self.num_features_metrical+self.num_features_categorical

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
    def num_features_metrical(self):
        return len(self.features_metrical)

    @property
    def num_features_categorical(self):
        if self.levels_categorical is None or self.features_categorical is None:
            return 0
        else:
            return sum([len(self.levels_categorical[feature]) for feature in self.features_categorical])


if __name__ == '__main__':
    pass
