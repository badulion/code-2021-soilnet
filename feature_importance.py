from cgi import test
import numpy as np
import hydra
from hydra.utils import get_original_cwd
from dataset.dataloader.labeledDS import LabeledDataModule, LabeledSoilData
from dataset.dataloader.unlabeledDS import UnlabeledDataModule
import os
from utils.metrics import AllMetrics
import json
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import trange
import pandas as pd
from model.SoilModel import SoilModel
from dataset.dataloader.patchDS import PatchDataModule


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg):
    feature_importances = pd.DataFrame({'fold': [], 'feature': [], 'importance': [], 'run': []})
    for fold in range(cfg.dataset.n_splits):

        train_ds = LabeledSoilData(
            path=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
            mode='train', fold=fold,
            features_metrical=cfg.vars.features_metrical,
            features_categorical=cfg.vars.features_categorical,
            levels_categorical=cfg.vars.levels_categorical,
            encoding_categorical=cfg.vars.encoding_categorical,
        )

        test_ds = LabeledSoilData(
            path=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
            mode='test', fold=fold,
            features_metrical=cfg.vars.features_metrical,
            features_categorical=cfg.vars.features_categorical,
            levels_categorical=cfg.vars.levels_categorical,
            encoding_categorical=cfg.vars.encoding_categorical,
        )


        model = SoilModel(cfg.model.name, cfg.model.parameters, train_ds.num_features, train_ds.num_data).model


        scaler = StandardScaler()
        train_ds.transform(scaler, fit=True)
        model.fit(*train_ds.get_data_as_np_array())


        test_ds.transform(scaler, fit=False)
        x_test, y_test = test_ds.get_data_as_np_array()

        pred = model.predict(x_test)
        baseline_mse = np.sum((pred-y_test)**2)

        feature_importance_list = []
        for id in range(test_ds.num_features):
            for run in range(100):
                x_test, y_test = test_ds.get_permuted_feature(id)
                pred = model.predict(x_test)
                permuted_mse = np.sum((pred-y_test)**2)
                feature_importance = (baseline_mse-permuted_mse)/baseline_mse
                feature_importance_dict = {
                    'fold': fold,
                    'feature': test_ds.get_feature_name(id),
                    'importance': feature_importance,
                    'run': run
                }
                feature_importance_list.append(feature_importance_dict)
        
        feature_importances_fold = pd.DataFrame(feature_importance_list)
        feature_importances = pd.concat([feature_importances, feature_importances_fold])


    feature_importances = feature_importances.reset_index()
    feature_importances_agg = feature_importances[['feature', 'importance']].groupby(['feature'], as_index=False).mean()

    feature_importances_grouped = feature_importances[['feature', 'importance']].groupby(['feature'])

    x = [group[1]['importance'].values for group in feature_importances_grouped]
    labels = [group[0] for group in feature_importances_grouped]

    import matplotlib.pyplot as plt 
    plt.boxplot(x, labels=labels)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)  # GPytorch bug triggers warnings
    my_app()
