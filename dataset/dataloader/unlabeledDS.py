import torch
import pandas as pd
import numpy as np
import os
import hydra
#import category_encoders as ce
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KernelDensity
import pytorch_lightning as pl
from model.SoilModel import SoilModel
from tqdm import tqdm


class UnlabeledSoilData(Dataset):
    """A dataset containing soil grid data for Lower, Middle and Upper Franconia."""

    def __init__(self,
                 path,
                 weak_model,
                 scaler,
                 labeled_depths,
                 mode: str = "train",
                 features_metrical=["x", "y"],
                 features_categorical=None,
                 levels_categorical=None,
                 encoding_categorical="onehot"):

        self.path = path

        assert mode in ["train", "val", "test"]
        self.mode = mode

        # save parameters
        self.features_metrical = features_metrical
        self.features_categorical = features_categorical
        self.levels_categorical = levels_categorical
        self.encoding_categorical = encoding_categorical
        self.scaler = scaler
        self.labeled_depths = labeled_depths
        self.weak_model = weak_model

        # prepare data
        self.data_iterator = self._get_iterator()
        self.cat_encoder = self._setup_cat_encoder()
        self.depth_sampler = self._setup_depth_sampler()
        self.x, self.y = self._setup_data()

    def _get_iterator(self):
        return pd.read_csv(self.path, chunksize=100000)

    def _setup_cat_encoder(self):
        cat_encoder = OneHotEncoder(categories=[self.levels_categorical[feature] for feature in self.features_categorical],
                                    sparse=False,
                                    dtype=np.float32)

        return cat_encoder

    def _setup_depth_sampler(self):
        depth_sampler = KernelDensity()
        depth_sampler.fit(self.labeled_depths)
        return depth_sampler

    def _setup_data(self):
        x = []
        y = []
        for chunk in self.data_iterator:
            x_batch, y_batch = self._process_chunk(chunk)
            x.append(x_batch)
            y.append(y_batch)
        return np.vstack(x), np.vstack(y)

    def _process_chunk(self, chunk):
        chunk["depth"] = self.depth_sampler.sample(len(chunk))

        # setup metrical
        chunk_metrical = chunk[self.features_metrical].values.astype(np.float32)
        chunk_metrical = self.scaler.transform(chunk_metrical)

        # setup categorical
        if self.features_categorical is None:
            chunk_categorical = np.empty((len(self.data), 0))
        else:
            chunk_categorical = chunk[self.features_categorical].values.astype(str)
            chunk_categorical = self.cat_encoder.fit_transform(chunk_categorical)

        chunk_x = np.hstack([chunk_metrical, chunk_categorical])
        chunk_y = self.weak_model.predict_x(chunk_x).astype(np.float32)
        return chunk_x, chunk_y

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


class UnlabeledDataModule(pl.LightningDataModule):
    def __init__(self,
                 path,
                 data_labeled,
                 weak_model,
                 mode='val',
                 fold=0,
                 num_splits=10,
                 batch_size=1024,
                 num_workers=1):
        super(UnlabeledDataModule, self).__init__()

        # save parameters
        self.mode = mode
        self.path = path
        self.fold = fold
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

        # prepare weak model
        self.weak_model = SoilModel(self.weak_model_dict.name,
                                    self.weak_model_dict.parameters,
                                    self.data_labeled.num_features)
        self.weak_model.fit(self.data_labeled)
        self.scaler = self.data_labeled.scaler
        self.labeled_depths = self.data_labeled.train.depths

        self.train = UnlabeledSoilData(
            path=self.path,
            weak_model=self.weak_model,
            scaler=self.scaler,
            labeled_depths=self.labeled_depths,
            features_metrical=self.features_metrical,
            features_categorical=self.features_categorical,
            levels_categorical=self.levels_categorical,
            encoding_categorical=self.encoding_categorical
        )

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
