"""
Useful functions involving statistical operations, called from various locations in the code.
"""
import os
import sys

import src.utilities.params as params  # get file location and varname parameters for

import numpy as np
import pandas as pd
from operator import itemgetter

# import statsmodels.api as sm

params.importIfNotAlready()


# I could also update this with the df.apply lambda method (see weighted_average function and application in preprocessing)
def weighted_mean_zonal(df, levels, weights):
    groups = np.sort(levels.unique()).astype(np.int64)
    df_l = pd.concat([levels, df], axis="columns")
    col = list(range(1, len(df_l.columns)))
    lists = [[] for g in range(0, len(groups))]
    for g in groups:
        df_g = df_l.loc[df_l.iloc[:, 0] == g]
        w_g = weights[df_g.index]
        for c in col:
            w_a = round(np.average(df_g.iloc[:, c], weights=w_g), 2)
            # here I need to append the list with the result
            lists[g - 1].append(w_a)
    results = pd.DataFrame(lists, index=[groups], columns=[df.columns])
    return results


# updated version for series, applicable to dfs via df.apply(lambda x:)
def weighted_average(data, weights, *, dropna: bool = False):
    df = pd.concat([data, weights], axis=1)
    if dropna == True:
        df = df.dropna()
    weight_mean = round(np.average(df.iloc[:, 0], weights=df.iloc[:, 1]), 2)
    return weight_mean


def weighted_mode(data, weights, *, dropna: bool = False):
    if dropna == True:
        data = data.dropna()
    categories = np.sort(data.unique())
    counts = dict.fromkeys(categories)
    for cat in categories:
        data_cat = data.loc[data == cat]
        if data_cat.empty:
            counts[cat] = 0
        else:
            weights_cat = weights[data_cat.index]
            weighted_count = weights_cat.sum()
            counts[cat] = weighted_count
    weight_mode = max(counts.items(), key=itemgetter(1))[0]
    return weight_mode
