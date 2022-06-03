import numpy as np
import hydra
from hydra.utils import get_original_cwd
from dataset.dataloader.labeledDS import LabeledDataModule
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
    predictions = []
    targets = []
    metric = AllMetrics()
    if cfg.model.name == 'soilcnn':
        for i in range(cfg.dataset.n_splits):

            data_labeled = LabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
                                             features_metrical=cfg.vars.features_metrical,
                                             features_categorical=cfg.vars.features_categorical,
                                             levels_categorical=cfg.vars.levels_categorical,
                                             encoding_categorical=cfg.vars.encoding_categorical,
                                             mode='test', fold=i)

            data_labeled_patch = PatchDataModule(path_lab=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
                                path_unlab=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),
                                n=cfg.patch.parameters.n,
                                deviation_to_shrink_df=cfg.patch.parameters.deviation_to_shrink_df,
                                deviation_for_perfect_hit1=cfg.patch.parameters.deviation_for_perfect_hit1,
                                deviation_for_perfect_hit2=cfg.patch.parameters.deviation_for_perfect_hit2,
                                deviation_between_two_points=cfg.patch.parameters.deviation_between_two_points,
                                features_metrical = cfg.vars.features_metrical,
                                features_categorical = cfg.vars.features_categorical,
                                mode="test")
                                
            data_unlabeled = UnlabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_weak_labeled),  
                                                 path_labeled=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),  
                                                 path_unlabeled=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),  
                                                 data_labeled=data_labeled,
                                                 weak_model=cfg.weak_model,
                                                 vars=cfg.vars.name,
                                                 fold=i)

            model = SoilModel(cfg.model.name, cfg.model.parameters, data_labeled_patch.num_features, data_labeled_patch.num_data)
            model.fit(data_labeled_patch, data_unlabeled)
            pred, y = model.predict(data_labeled_patch)
            metric.update(pred, y)

            predictions.append(pred)
            targets.append(y)
    else:
        for i in range(cfg.dataset.n_splits):
            data_labeled = LabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
                                             features_metrical=cfg.vars.features_metrical,
                                             features_categorical=cfg.vars.features_categorical,
                                             levels_categorical=cfg.vars.levels_categorical,
                                             encoding_categorical=cfg.vars.encoding_categorical,
                                             mode='test', fold=i)

            data_unlabeled = UnlabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_weak_labeled),  
                                                 path_labeled=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),  
                                                 path_unlabeled=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),  
                                                 data_labeled=data_labeled,
                                                 weak_model=cfg.weak_model,
                                                 vars=cfg.vars.name,
                                                 fold=i)
            data_unlabeled=None
            model = SoilModel(cfg.model.name, cfg.model.parameters, data_labeled.num_features, data_labeled.num_data)
            model.fit(data_labeled, data_unlabeled)
            pred, y = model.predict(data_labeled)
            metric.update(pred, y)

            predictions.append(pred)
            targets.append(y)

    results = metric.calculate()
    model_results = {'model': cfg.model.name}
    model_results.update(results)
    with open('results.json', mode='w') as file:
        json.dump(model_results, file)

    if cfg.general.verbose:
        print(metric.calculate_string())

    if cfg.general.save_predictions:
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        pred_target = np.hstack([targets, predictions])
        columns = cfg.vars.targets + [t + "_pred" for t in cfg.vars.targets]
        df = pd.DataFrame(data=pred_target, columns=columns)
        df.to_csv("predictions.csv", index=False)

    return results['mae']


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)  # GPytorch bug triggers warnings
    my_app()
