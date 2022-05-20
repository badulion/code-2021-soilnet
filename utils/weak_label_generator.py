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
from dataset.dataloader.labeledDS import LabeledDataModule


class WeakLabelGenerator:
    """A dataset containing soil grid data for Lower, Middle and Upper Franconia."""

    def __init__(self,
                 path_labeled,
                 path_unlabeled,
                 path_output,
                 weak_model_dict,
                 mode: str = "train",
                 fold: int = 0,
                 depth_mode: str = "distribution",
                 constant_depth_level=None,
                 features_metrical=["x", "y"],
                 features_categorical=None,
                 levels_categorical=None,
                 targets=['sand', 'silt', 'clay'],
                 encoding_categorical="onehot"):

        self.path_unlabeled = path_unlabeled
        self.path_labeled = path_labeled
        self.path_output = path_output

        assert mode in ["train", "val", "test", "all"]
        self.mode = mode
        self.depth_mode = depth_mode

        # save parameters
        self.features_metrical = features_metrical
        self.features_categorical = features_categorical
        self.levels_categorical = levels_categorical
        self.encoding_categorical = encoding_categorical
        self.constant_depth_level = constant_depth_level
        self.targets = targets

        # prepare labeled data
        self.data_labeled = LabeledDataModule(
            path=path_labeled,
            features_metrical=features_metrical,
            features_categorical=features_categorical,
            levels_categorical=levels_categorical,
            encoding_categorical=encoding_categorical,
            mode=mode,
            fold=fold
        )
        self.data_labeled.setup()
        self.labeled_depths = self.data_labeled.train.depths

        # prepare weak model
        self.weak_model = SoilModel(weak_model_dict.name,
                                    weak_model_dict.parameters,
                                    self.data_labeled.num_features,
                                    self.data_labeled.num_data)
        
        self.weak_model.fit(self.data_labeled)

        # prepare data
        self.data_iterator = self._get_iterator()
        self.scaler = self.data_labeled.scaler
        self.cat_encoder = self._setup_cat_encoder()
        self.depth_sampler = self._setup_depth_sampler()

    def _get_iterator(self):
        return pd.read_csv(self.path_unlabeled, chunksize=10000)

    def _setup_cat_encoder(self):
            if not self.features_categorical:
                return None
            else:
                cat_encoder = OneHotEncoder(categories=[self.levels_categorical[feature] for feature in self.features_categorical],
                                    sparse=False,
                                    dtype=np.float32)
            return cat_encoder

    def _setup_depth_sampler(self):
        if self.depth_mode == "distribution":
            depth_sampler = KernelDensity()
            depth_sampler.fit(self.labeled_depths)
            return depth_sampler
        else:
            class ConstantDepthSampler:
                def __init__(self, depth) -> None:
                    self.depth = depth
           
                def sample(self, len: int):
                    return np.repeat(self.depth, len)
            return ConstantDepthSampler(self.constant_depth_level)

    def generate(self):
        if not os.path.exists(os.path.dirname(self.path_output)):
            os.makedirs(os.path.dirname(self.path_output))
        i = 0
        if os.path.exists(self.path_output):
            os.remove(self.path_output)
        for chunk in tqdm(self.data_iterator):
            chunk_pred = self._predict_chunk(chunk)
            chunk_pred.to_csv(self.path_output, mode='a', header=not os.path.exists(self.path_output))
            if False:
                break
            i += 1

    def get_length(self, unit='batches'):
        ds_len = 0
        for chunk in self.data_iterator.copy():
            if unit == 'batches':
                ds_len += 1
            else:
                ds_len += len(chunk)
        return ds_len

    def _predict_chunk(self, chunk):
        chunk["depth"] = self.depth_sampler.sample(len(chunk))

        # setup metrical
        chunk_metrical = chunk[self.features_metrical].values.astype(np.float32)
        chunk_metrical = self.scaler.transform(chunk_metrical)

        # setup categorical
        if not self.features_categorical:
            chunk_categorical = np.empty((len(chunk_metrical), 0))
        else:
            chunk_categorical = chunk[self.features_categorical].values.astype(str)
            chunk_categorical = self.cat_encoder.fit_transform(chunk_categorical)

        chunk_x = np.hstack([chunk_metrical, chunk_categorical])
        pred = self.weak_model.predict_x(chunk_x).astype(np.float32)
        chunk[self.targets] = pred
        return chunk
