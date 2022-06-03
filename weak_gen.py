import hydra
import os
from hydra.utils import get_original_cwd

from utils.weak_label_generator import WeakLabelGenerator


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg):
    for fold in range(cfg.dataset.n_splits):
        data_labeled = WeakLabelGenerator(
            path_labeled=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
            path_unlabeled=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),
            path_output=os.path.join(get_original_cwd(),f"dataset/data/Feb_2022/weak_labels/{cfg.vars.name}/fold_{fold}.csv"),
            weak_model_dict = cfg.model,
            mode = "all",
            fold = fold,
            depth_mode = "distribution",
            constant_depth_level=None,
            features_metrical=cfg.vars.features_metrical,
            features_categorical=cfg.vars.features_categorical,
            levels_categorical=cfg.vars.levels_categorical,
            encoding_categorical=cfg.vars.encoding_categorical
        )
        data_labeled.generate()

if __name__ == '__main__':
    my_app()
