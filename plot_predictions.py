import hydra
import os
from hydra.utils import get_original_cwd

from utils.weak_label_generator import WeakLabelGenerator
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from sklearn.metrics import r2_score

from geocube.api.core import make_geocube

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils.metrics import soil_classification
from utils.geometric_median import geometric_median
from utils.plots import draw_borders

color_dict = {
    'clay': '#dc342f',
    'silty clay': '#b8689e',
    'sandy clay': '#e3622a',
    'clay loam': '#3d2314',
    'silty clay loam': '#564839',
    'sandy clay loam': '#a66228',
    'loam': '#72665a',
    'silt loam': '#9c8679',
    'silt': '#9294a1',
    'sandy loam': '#a09160',
    'loamy sand': '#dbaf20',
    'sand': '#dbc200',
}

class_dict = {
    0: 'clay',
    1: 'silty clay',
    2: 'sandy clay',
    3: 'clay loam',
    4: 'silty clay loam',
    5: 'sandy clay loam',
    6: 'loam',
    7: 'silt loam',
    8: 'silt',
    9: 'sandy loam',
    10: 'loamy sand',
    11: 'sand',
}

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/256. for i in range(0, lv, lv // 3))

class_list = list(color_dict.keys())
class_list.sort()
soil_cmap = ListedColormap([hex_to_rgb(color_dict[c]) for c in class_list])


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg):
    if False:
        zip_file = os.path.join("zip:", get_original_cwd(), "dataset/data/county/cb_2013_us_county_20m.shp")
        counties = gpd.read_file(zip_file)
        counties["LSAD_NUM"] = counties.LSAD.astype(int)
        cube = make_geocube(
            counties,
            measurements=["LSAD_NUM"],
            resolution=(1, -1),
        )
        print(cube['LSAD_NUM'])
        
        cube.LSAD_NUM.plot()
        plt.show()
        quit()

    for depth in [4,20,50,100, 30000]:
        data_labeled = WeakLabelGenerator(
            path_labeled=os.path.join(get_original_cwd(), cfg.dataset.path_labeled),
            path_unlabeled=os.path.join(get_original_cwd(), cfg.dataset.path_unlabeled),
            path_output=os.path.join(get_original_cwd(),f"dataset/data/Feb_2022/predictions/{cfg.vars.name}/depth_{depth}.csv"),
            weak_model_dict = cfg.weak_model,
            mode = "all",
            fold = 0,
            depth_mode = "constant",
            constant_depth_level=depth,
            features_metrical=cfg.vars.features_metrical,
            features_categorical=cfg.vars.features_categorical,
            levels_categorical=cfg.vars.levels_categorical,
            encoding_categorical=cfg.vars.encoding_categorical
        )
        data_labeled.generate()
        
        #read the generated data
        weak_labels_path = os.path.join(get_original_cwd(),f"dataset/data/Feb_2022/predictions/{cfg.vars.name}/depth_{depth}.csv")
        df = pd.read_csv(weak_labels_path)
        
        df["soil_class_id"] = soil_classification(df[["sand","silt","clay"]].values)
        df["soil_class"] = [class_dict[soil] for soil in df["soil_class_id"]]
        df = df.sort_values(by="soil_class_id")



        #generate geometries
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
        #gdf.crs = "epsg:32632"
        gdf.crs = "epsg:25832"

        #making geocube
        variables = cfg.vars.features_metrical + \
                    cfg.vars.features_categorical + \
                    ['sand', 'silt','clay','soil_class_id']

        variables = filter(lambda var: var not in ["x", "y", "depth"], variables)

        cube = make_geocube(
            gdf,
            measurements=variables,
            resolution=(-50,50),
            #categorical_enums={'soil_class': list(class_dict.values())},
        )

        # restore categorical data
        #cube['soil_class_id'] = cube['soil_class'].where(cube['soil_class'] != -1)
        #cube['soil_class'] = cube['soil_class_categories'][cube['soil_class'].astype(int)]

        #create dir where objects are saved
        save_path = os.path.join(get_original_cwd(), f"results/vis/{cfg.vars.name}/")
        os.makedirs(save_path, exist_ok=True)

        # save raster
        raster_path = os.path.join(save_path, f"raster_depth_{depth}.tif")
        cube.rio.to_raster(raster_path)

        #save netcfs
        netcfs_path = os.path.join(save_path, f"netcfs_depth_{depth}.nc")
        cube.to_netcdf(netcfs_path)

        #draw predictions
        reg_bez_path = os.path.join(get_original_cwd(),f"dataset/data/reg_bez/reg_bez.shp")
        fig, ax = draw_borders(reg_bez_path)

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y).buffer(25, cap_style=3))
        gdf.crs = "epsg:25832"
        gdf = gdf.to_crs(epsg=4326)
        gdf.plot(ax=ax, column="soil_class", cmap=soil_cmap, legend=True)

        #save figure
        fig_path = os.path.join(save_path, f"figure_depth_{depth}.png")
        fig.savefig(fig_path)
    

if __name__ == '__main__':
    my_app()

