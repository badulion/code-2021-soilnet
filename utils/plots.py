import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def draw_borders(shapefile="dataset/data/reg_bez/reg_bez.shp"):
    bez = gpd.read_file(shapefile)
    
    fig, ax = plt.subplots(figsize=(20,16))
    ax.set_aspect('equal')
    fig.set_dpi(120)
    bez.plot(ax=ax, color="none", alpha=0.8, edgecolor='black', linewidth=2)
    ax.grid()
    ax.set_xticks(ticks=np.arange(9.0, 12.5, 0.5), labels=["9° 0' E", "9° 30' E", "10° 0' E", "10° 30' E", "11° 0' E", "11° 30' E", "12° 0' E"])
    ax.set_yticks(ticks=np.arange(49.0, 50.75, 0.25), labels=["49° 0' N", "49° 15' N", "49° 30' N", "49° 45' N", "50° 0' N", "50° 15' N", "50° 30' N"])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    return fig, ax



if __name__== "__main__":
    shapefile="dataset/data/reg_bez_original/reg_bez.shp"
    bez = gpd.read_file(shapefile)
    bez.crs = "epsg:4326"
    os.makedirs("dataset/data/reg_bez", exist_ok=True)
    bez.to_file("dataset/data/reg_bez/reg_bez.shp")
