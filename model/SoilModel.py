import numpy as np

# baseline models:
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor
from model.ExactGP import ExactGPModel
from model.VariationalGP import VariationalGP
from model.weightedIDW import WeightedIDWModel
from model.SKLearnGP import SKLearnGPModel
from model.SoilNet import SoilMLP, SoilNet


class SoilModel:
    def __init__(self, name, params, num_features, num_data):
        self.name = name
        if name == 'catboost':
            self.model = CatBoostRegressor(**params)
        elif name == 'rf':
            self.model = RandomForestRegressor(**params)
        elif name == 'svm':
            self.model = MultiOutputRegressor(SVR(**params))
        elif name == 'idw':
            self.model = KNeighborsRegressor(**params)
        elif name == 'weighted_idw':
            self.model = WeightedIDWModel(num_features=num_features, **params)
        elif name == 'knn':
            self.model = KNeighborsRegressor(**params)
        elif name == 'sklearn_gp':
            self.model = SKLearnGPModel(**params)
        elif name == 'exact_gp':
            self.model = ExactGPModel(num_features=num_features, **params)
        elif name == 'vargp':
            self.model = VariationalGP(num_features=num_features, num_data=num_data, **params)
        elif name == 'mlp':
            self.model = SoilMLP(num_features=num_features, **params)
        elif name == 'soilnet':
            self.model = SoilNet(num_features=num_features, **params)
        else:
            self.model = DummyRegressor()

    def fit(self, train_datamodule, pretrain_datamodule=None):
        if not train_datamodule.is_setup:
            train_datamodule.setup()
        if self.name == 'mlp':
            self.model.fit(train_datamodule)
        elif self.name == 'vargp':
            self.model.fit(train_datamodule)
        elif self.name == "soilnet":
            self.model.fit(train_datamodule, pretrain_datamodule)
        else:
            train_ds = train_datamodule.train
            self.model.fit(*train_ds.get_data_as_np_array())

    def predict(self, train_datamodule, pretrain_datamodule=None):
        if not train_datamodule.is_setup:
            raise Exception("The datamodule has not been initialized. Did you fit the model?")

        test_ds = train_datamodule.test
        X, y = test_ds.get_data_as_np_array()

        if self.name in ['mlp', 'vargp', 'soilnet']:
            preds = self.model.predict(train_datamodule)
        else:
            preds = self.model.predict(X)
        return self.normalize_predictions(preds), y

    def predict_x(self, X):
        if self.name == 'mlp' or self.name == "soilnet":
            raise Exception("Only baseline models can be used with this prediction type")

        preds = self.model.predict(X)
        return self.normalize_predictions(preds)

    def normalize_predictions(self, pred):
        pred = np.clip(pred, 0.0, 1.0)
        return pred/np.sum(pred, axis=1, keepdims=True)
