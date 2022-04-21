<div id="top"></div>

1. Was ist dieses Repo
2. Modelle auflisten und kurz beschreiben
3. Prerequisites
4. How to get Started
5. Running Experiments

<!-- ABOUT THE PROJECT -->
## About The Project

This is the current code for the newest SoilNet models and experiments.  
You can run your own experiments using different models with this easily.

### Models
There are a variety of models you can choose from.  
Just execute the code with the model you want (see "usage")  
This is a short explanation of the models:
#####convolutional Neural Network (model=soilcnn)  
creates a cnn, existing of a a few convolutional Layers including a max-pool,  
followed by a fully connected layer.  #zitat
#####CatBoostRegressor (model=catboost)
#####RandomForestRegressor (model=rf)
#####Epsilon-Support Vector Regression (model=svm)
#####KNeighborsRegressor (model=idw)
#####WeightedIDW (model=weighted_idw)
#####KNeighborsRegressor (model=knn)
#####SKLearnGP (model=sklearn_gp)
#####ExactGP (model=exact_gp)
#####VariationalGP (model=vargp)
#####Multi-layer Perceptron (MLP) (model=mlp)
#####Neuronal Network (model=soilnet)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This describes how to setup the code.

### Prerequisites

To get started you will need a python3 environment and the soil dataset. Copy both the labeled and unlabeled datasets to `dataset/data/%dataset name%` and change the appropriate paths in `config.yaml`

### Installation

To run the experiments you will need the packages listed in `requirements.txt`

install them using
```sh
pip install -r requirements.txt
```
or any other method of your choice.



<!-- USAGE EXAMPLES -->
## Usage

To run the experiments simply run the `main.py` script, overriding the configurations if necessary using hydra syntax, e.g.:

```
python3 main.py model=idw vars=metrical
```

### Adjusting code for different models
#####running the convolutional neural network model
In order to execute 
You can config attributes of the patches in `conf/patch/make_patch`
Importent attributes are n for the patch size (n=5 results in 11x11 patches) and deviation_between_two_points to set up how much deviation there is between two points.

In order to run the convolutional neural network model the loop starting in line 21 need to look like this:
```
    for i in range(cfg.dataset.n_splits):

        data_labeled = LabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
                                         features_metrical=cfg.vars.features_metrical,
                                         features_categorical=cfg.vars.features_categorical,
                                         levels_categorical=cfg.vars.levels_categorical,
                                         encoding_categorical=cfg.vars.encoding_categorical,
                                         mode='val', fold=i)

        data_labeled_patch = PatchDataModule(path_lab=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
                            path_unlab=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),
                            n=cfg.patch.parameters.n,
                            deviation_to_shrink_df=cfg.patch.parameters.deviation_to_shrink_df,
                            deviation_for_perfect_hit1=cfg.patch.parameters.deviation_for_perfect_hit1,
                            deviation_for_perfect_hit2=cfg.patch.parameters.deviation_for_perfect_hit2,
                            deviation_between_two_points=cfg.patch.parameters.deviation_between_two_points,
                            features_metrical = cfg.vars.features_metrical,
                            features_categorical = cfg.vars.features_categorical,
                            mode="val")
        data_unlabeled = UnlabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),
                                             data_labeled=data_labeled,
                                             weak_model=cfg.weak_model,
                                             mode='val', fold=i)
        model = SoilModel(cfg.model.name, cfg.model.parameters, data_labeled_patch.num_features, data_labeled_patch.num_data)
        model.fit(data_labeled_patch, data_unlabeled)
        pred, y = model.predict(data_labeled_patch)
        metric.update(pred, y)

        predictions.append(pred)
        targets.append(y)
        
```
#####running the SoilNet model
In order to run the SoilNet model the loop staring in line 21 need to look like this:
```
    for i in range(cfg.dataset.n_splits):
        data_labeled = LabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
                                         features_metrical=cfg.vars.features_metrical,
                                         features_categorical=cfg.vars.features_categorical,
                                         levels_categorical=cfg.vars.levels_categorical,
                                         encoding_categorical=cfg.vars.encoding_categorical,
                                         mode='val', fold=i)

        data_unlabeled = UnlabeledDataModule(path=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),
                                             data_labeled=data_labeled,
                                             weak_model=cfg.weak_model,
                                             mode='val', fold=i)

        model = SoilModel(cfg.model.name, cfg.model.parameters, data_labeled.num_features, data_labeled.num_data)
        model.fit(data_labeled, data_unlabeled)
        pred, y = model.predict(data_labeled)
        metric.update(pred, y)

        predictions.append(pred)
        targets.append(y)
        
```




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
