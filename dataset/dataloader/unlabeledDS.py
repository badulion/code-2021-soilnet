import torch
import pandas as pd
import numpy as np
import os
import hydra
import category_encoders as ce
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KernelDensity
import pytorch_lightning as pl
from model.SoilModel import SoilModel
from tqdm import tqdm
from utils.weak_label_generator import WeakLabelGenerator


class UnlabeledSoilData(Dataset):
    """A dataset containing soil grid data for Lower, Middle and Upper Franconia."""

    def __init__(self,
                 path,
                 path_labeled,
                 path_unlabeled,
                 weak_model_dict,
                 vars: str = "full",
                 fold: int = 0,
                 mode: str = "train",
                 features_metrical=["x", "y"],
                 features_categorical=None,
                 levels_categorical=None,
                 encoding_categorical="onehot",
                 targets=["sand", "silt", "clay"],
                 bezirk_column="Bezirk",
                 bezirk=None,
                 frac=0.1,
                 ):

        self.path = path
        self.path_labeled = path_labeled
        self.path_unlabeled = path_unlabeled
        self.fold = fold
        self.vars = vars
        self.weak_model_dict = weak_model_dict

        assert mode in ["train", "val", "test", "all"]
        self.mode = mode

        # save parameters
        self.features_metrical = features_metrical
        self.features_categorical = features_categorical
        self.levels_categorical = levels_categorical
        self.encoding_categorical = encoding_categorical
        self.targets = targets
        self.bezirk = bezirk
        self.bezirk_column = bezirk_column
        self.frac = frac

        self.train_val_test = np.cumsum([0.8, 0.1, 0.1])

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


    def _setup_data(self, fold):
        data_path = os.path.join(self.path, self.vars, self.weak_model_dict.name, f"fold_{fold}.csv")
        if not  os.path.exists(data_path):
            print("Weak labels not present. Generating dataset first...")
            self._generate_weak_labels(data_path)
        data = pd.read_csv(data_path)
        data = data.sample(frac=self.frac, random_state=42)

        train_index = int(len(data)*self.train_val_test[0])
        val_index = int(len(data)*self.train_val_test[1])
        test_index = int(len(data)*self.train_val_test[2])

        if self.mode == "train":
            data = data.iloc[:train_index]
        elif self.mode == "val":
            data = data.iloc[train_index:val_index]
        elif self.mode == "test":
            data = data.iloc[val_index:test_index]

        if self.bezirk is not None:
            return data.loc[[b in self.bezirk for b in data[self.bezirk_column]], :]
        else:
            return data

    def _generate_weak_labels(self, path):

        data_labeled = WeakLabelGenerator(
            path_labeled=self.path_labeled,
            path_unlabeled=self.path_unlabeled,
            path_output=path,
            weak_model_dict = self.weak_model_dict,
            mode = "train",
            fold = self.fold,
            depth_mode = "distribution",
            constant_depth_level=None,
            features_metrical=self.features_metrical,
            features_categorical=self.features_categorical,
            levels_categorical=self.levels_categorical,
            encoding_categorical=self.encoding_categorical
        )
        data_labeled.generate()

    def _setup_targets(self):
        targets = self.data[self.targets].values.astype(np.float32)
        return targets / np.sum(targets, axis=1, keepdims=True)

    def _setup_cat_features(self):
        if not self.features_categorical:
            return np.empty((len(self.data), 0), dtype=np.float32)
        else:
            x_categorical = self.data[self.features_categorical].values.astype(str)

        return x_categorical

    def _setup_metric_features(self):
        x_metric = self.data[self.features_metrical].values.astype(np.float32)
        return x_metric

    def transform(self, scaler=None, fit=False):
        if scaler == None:
            scaler = self._create_scaler()

        func = scaler.fit_transform if fit else scaler.transform
        self.x_metric = func(self.x_metric)

        if not self.features_categorical:
            self.x_categorical = np.empty((len(self.data), 0), dtype=np.float32)
        else:
            self.cat_encoder = OneHotEncoder(categories=[self.levels_categorical[feature] for feature in self.features_categorical],
                                             sparse=False,
                                             dtype=np.float32)

            self.x_categorical = self.cat_encoder.fit_transform(self.x_categorical)

        self.x = np.hstack([self.x_metric, self.x_categorical])

    def _create_scaler(self):
        scaler = StandardScaler()
        return scaler

    def __getitem__(self, i: int) -> tuple:
        x = self.x[i]
        y = self.y[i]

        return x, y

    def __len__(self) -> int:
        return len(self.x)

    def inverse_transform(self, scaler):
        self.x_metric = scaler.inverse_transform(self.x_metric)

        if self.features_categorical is None:
            self.x_categorical = np.empty((len(self.data), 0), dtype=np.float32)
        else:
            self.x_categorical = self.cat_encoder.inverse_transform(self.x_categorical)

        self.x = np.hstack([self.x_metric, self.x_categorical])

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


class UnlabeledDataModule(pl.LightningDataModule):
    def __init__(self,
                 path,
                 path_labeled,
                 path_unlabeled,
                 data_labeled,
                 weak_model,
                 vars,
                 fold=0,
                 num_splits=10,
                 batch_size=1024,
                 num_workers=1):
        super(UnlabeledDataModule, self).__init__()

        # save parameters
        self.path = path
        self.path_labeled = path_labeled
        self.path_unlabeled = path_unlabeled
        self.fold = fold
        self.vars = vars
        self.weak_model_dict = weak_model
        self.data_labeled = data_labeled
        self.features_metrical = data_labeled.features_metrical
        self.features_categorical = data_labeled.features_categorical
        self.levels_categorical = data_labeled.levels_categorical
        self.encoding_categorical = data_labeled.encoding_categorical
        self.targets = data_labeled.targets
        self.num_splits = num_splits
        self.batch_size = batch_size
        self.num_workers = num_workers

        #internal setup
        self._is_setup = False

    def setup(self, stage=None):

        self.scaler = self.data_labeled.scaler

        self.train = UnlabeledSoilData(
            path=self.path,
            path_labeled=self.path_labeled,
            path_unlabeled=self.path_unlabeled,
            weak_model_dict=self.weak_model_dict,
            vars=self.vars,
            fold=self.fold,
            mode="train",
            features_metrical=self.features_metrical,
            features_categorical=self.features_categorical,
            levels_categorical=self.levels_categorical,
            encoding_categorical=self.encoding_categorical,
            frac=0.1,
        )
        """
        self.val = UnlabeledSoilData(
            path=self.path,
            weak_model_dict=self.weak_model_dict,
            vars=self.vars,
            fold=self.fold,
            mode="val",
            features_metrical=self.features_metrical,
            features_categorical=self.features_categorical,
            levels_categorical=self.levels_categorical,
            encoding_categorical=self.encoding_categorical,
            frac=0.1,
        )
        """

        self.train.transform(self.scaler, fit=False)
        #self.val.transform(self.scaler, fit=False)
        self._is_setup = True

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_labeled.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_labeled.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.data_labeled.test, batch_size=self.batch_size, num_workers=self.num_workers)

    @ property
    def is_setup(self):
        return self._is_setup

    @ property
    def num_features(self):
        return self.num_features_metrical+self.num_features_categorical

    @ property
    def num_features_metrical(self):
        return len(self.features_metrical)

    @ property
    def num_features_categorical(self):
        if self.levels_categorical is None or self.features_categorical is None:
            return 0
        else:
            return sum([len(self.levels_categorical[feature]) for feature in self.features_categorical])


if __name__ == '__main__':
    pass
