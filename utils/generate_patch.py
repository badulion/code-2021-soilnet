import pandas as pd
import numpy as np
from tqdm import trange
import hydra
import os
from hydra.utils import get_original_cwd
from dataset import data

class Patchgenerator():
    def __init__(self, df_lab, df_unlab, n, amount_params, deviation_to_shrink_df, deviation_for_perfect_hit1, deviation_for_perfect_hit2, deviation_between_two_points):
        self.df_lab = df_lab
        self.df_unlab = df_unlab
        self.n = n * 2 + 1
        self.amount_params = amount_params
        self.deviation_to_shrink_df = deviation_to_shrink_df
        self.deviation_for_perfect_hit1 = deviation_for_perfect_hit1
        self.deviation_for_perfect_hit2 = deviation_for_perfect_hit2
        self.deviation_between_two_points = deviation_between_two_points

    def generate_patch(self):
        punkte_x, punkte_y = self.create_points(self.df_lab)
        if len(punkte_x) != len(punkte_y):
            print("Error, falscher Input")
            return
        else:
            result_arrays = []
            for i in trange(len(punkte_x)):
                    result_arrays.append(self.find_patch(punkte_x[i], punkte_y[i]))
            result = np.stack(result_arrays)
        return result

    def find_patch(self, x, y):
        df_customized = self.reduce_df(self.deviation_to_shrink_df, x, y, self.n, self.df_unlab)
        bester_punkt = self.find_best_point(df_customized, x, y)
        df_result = self.find_best_patch(df_customized, bester_punkt)
        return df_result

    def reduce_df(self, deviation_to_shrink_df, x, y, n, df):
        max_deviation = deviation_to_shrink_df * n
        df_new = df.loc[(df['x'] < x + max_deviation) & (df['y'] < y + max_deviation) & (df['x'] > x - max_deviation) & (df['y'] > y - max_deviation)]
        return df_new

    def create_points(self, df):
        punkte_x = df['x'].to_numpy().tolist()
        punkte_y = df['y'].to_numpy().tolist()
        return punkte_x, punkte_y

    def create_2D_list(self, n):
        result = []
        for i in range(n):
            result.append([])
            for k in range(n):
                result[i].append(0)
        return result

    def find_next_spot(self, x, y, direction):
        if direction == 0: #east
            x = x + 1
        elif direction == 1: #south
            y = y - 1
        elif direction == 2: #west
            x = x - 1
        elif direction == 3: #north
            y = y + 1
        return x, y

    def rotate_direction(self, direction):
        if direction == 3:
            return 0
        else:
            return (direction + 1)

    def find_best_patch(self, df, bester_punkt):
        result = self.create_2D_list(self.n)
        if self.n == 1:
            return result
        x = self.n // 2  # starts in the center
        y = self.n // 2  # starts in the center
        direction = 0  # 0 = east, 1 = south, 2 = west, 3 = north
        current_point = bester_punkt
        result[x][y] = bester_punkt.to_numpy()
        for i in range(self.n - 1):
            next_point = self.find_next_point(df, current_point, direction)
            x, y = self.find_next_spot(x, y, direction)
            result[x][y] = next_point.to_numpy()
            current_point = next_point
            direction = self.rotate_direction(direction)
            for k in range(i + 1):
                next_point = self.find_next_point(df, current_point, direction)
                x, y = self.find_next_spot(x, y, direction)
                result[x][y] = next_point.to_numpy()
                current_point = next_point
            direction = self.rotate_direction(direction)
            for k in range(i + 1):
                next_point = self.find_next_point(df, current_point, direction)
                x, y = self.find_next_spot(x, y, direction)
                result[x][y] = next_point.to_numpy()
                current_point = next_point
        nump_result = np.array(result)
        nump_result = np.moveaxis(nump_result, 0, -1)
        nump_result = np.moveaxis(nump_result, 0, -1)
        return nump_result

    def find_next_point(self, df, current_point, direction):
        x = current_point['x']
        y = current_point['y']
        if direction == 0: #east
            result = self.find_best_point(df, x, y + self.deviation_between_two_points)
        elif direction == 1: #south
            result = self.find_best_point(df, x - self.deviation_between_two_points, y)
        elif direction == 2: #west
            result = self.find_best_point(df, x, y - self.deviation_between_two_points)
        elif direction == 3: #north
            result = self.find_best_point(df, x + self.deviation_between_two_points, y)
        return result

    def find_best_point(self, df, x, y):
        # Dataframe is reduced to a few points, optimally to one point. If it is only one point, return that point as result.
        df_few_datapoints = df.loc[(df['x'] < x + self.deviation_for_perfect_hit1) &
                                       (df['y'] < y + self.deviation_for_perfect_hit1) &
                                       (df['x'] > x - self.deviation_for_perfect_hit1) &
                                       (df['y'] > y - self.deviation_for_perfect_hit1)]
        if df_few_datapoints.shape == (1, self.amount_params):
            return df_few_datapoints.iloc[0]
        else:
            #Dataframe is reduced to even fewer points, optimally to one point. If it is only one point, return that point as result.
            df_onepoint = df.loc[(df['x'] < x + self.deviation_for_perfect_hit2) &
                                       (df['y'] < y + self.deviation_for_perfect_hit2) &
                                       (df['x'] > x - self.deviation_for_perfect_hit2) &
                                       (df['y'] > y - self.deviation_for_perfect_hit2)]
        if df_onepoint.shape == (1, self.amount_params):
            return df_onepoint.iloc[0]
        # if there are no points in df_onepoint, use df_few_datapoints to get the closest point
        if df_few_datapoints.shape == (0, self.amount_params):
            df_new = df
        else:
            df_new = df_few_datapoints
        test = True
        result = 0
        result_score = 99999
        for i in range(len(df_new)):
            if (abs(df_new.iloc[i]['x'] - x) + abs(df_new.iloc[i]['y'] - y)) < result_score:
                test = False
                result = df_new.iloc[i]
                result_score = abs(df_new.iloc[i]['x'] - x) + abs(df_new.iloc[i]['y'] - y)
        if test:
            # Error, no best point was found
            print("Error on point (x: " + str(x) + " y: " + str(y) + ")")
        return result
