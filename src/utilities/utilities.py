"""

Useful functions that don't involve plotting, called from various locations in the code.

"""
import src.utilities.params as params  # get file location and varname parameters for

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely

params.importIfNotAlready()


# this function is used to make a bunch of rectangle grid shapes so the
# plotting looks nice and so we can later add up the crop area inside the grid
def makeGrid(df):
    cell_size_lats = params.latdiff
    cells = []
    for index, row in df.iterrows():
        cell = shapely.geometry.Point(
            [
                row["lons"],
                row["lats"],
                row["lons"] + params.londiff,
                row["lats"] + cell_size_lats,
            ]
        ) 
        cells.append(cell)
    crs = {"init": "epsg:4326"}
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=cells)
    geo_df = geo_df.sort_values(by=["lats", "lons"])
    geo_df = geo_df.reset_index(drop=True)
    return geo_df


# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
# downsample the 2d array so that crop percentages are averaged.
def rebin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
# downsample the 2d array so that crop percentages are averaged.
def rebinIgnoreZeros(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    asmall = a.reshape(sh).mean(-1).mean(1)

    # when zero, true, false otherwise. Number gives fraction of cells that are
    # nonzero
    anonzeros = 1 - (a == 0).reshape(sh).mean(-1).mean(1)

    # if all cells are zero, report nan for yield
    anonzeronan = np.where(anonzeros == 0, np.nan, anonzeros)

    return np.divide(asmall, anonzeronan)


# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
# downsample the 2d array, but add all the values together.
def rebinCumulative(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    asmall = a.reshape(sh).mean(-1).mean(1)
    product = np.product((np.array(a.shape) / np.array(shape)))
    return asmall * product


# save a .pkl file with the gridded data saved in columns labelled by month
# number
# (save the imported nuclear winter data in geopandas format)
def saveasgeopandas(name, allMonths, gridAllMonths, lats, lons):
    assert len(allMonths) == len(gridAllMonths)

    # create 2D arrays from 1d latitude, longitude arrays
    lats2d, lons2d = np.meshgrid(lats, lons)

    data = {"lats": pd.Series(lats2d.ravel()), "lons": pd.Series(lons2d.ravel())}

    for i in range(0, len(allMonths)):
        grid = gridAllMonths[i]
        month = allMonths[i]
        data[month] = pd.Series(grid.ravel())
    df = pd.DataFrame(data=data)
    geometry = gpd.points_from_xy(df.lons, df.lats)
    gdf = gpd.GeoDataFrame(df, crs={"init": "epsg:4326"}, geometry=geometry)

    grid = makeGrid(gdf)

    grid = grid.sort_values(by=["lats", "lons"])
    return grid


# save a .pkl file with the gridded data saved in columns labelled by month
# number
def saveDictasgeopandas(name, data):
    df = pd.DataFrame(data)
    geometry = gpd.points_from_xy(df.lons, df.lats)
    gdf = gpd.GeoDataFrame(df, crs={"init": "epsg:4326"}, geometry=geometry)
    grid = makeGrid(gdf)
    fn = params.geopandasDataDir + name + ".pkl"

    grid = grid.sort_values(by=["lats", "lons"])
    grid.to_pickle(fn)

    return grid


# create a global ascii at arbitrary resolution
def createASCII(df, column, fn):
    # set creates a list of unique values
    cellsizelats = 180 / len(set(df["lats"]))
    cellsizelons = 360 / len(set(df["lons"]))
    assert cellsizelats == cellsizelons
    print("cellsizelats")
    print(cellsizelats)
    print("cellsizelons")
    print(cellsizelons)
    file1 = open(fn + ".asc", "w")  # write mode
    ncols = len(set(df["lons"]))
    nrows = len(set(df["lats"]))
    array = np.array(df[column]).astype("float32")
    arrayWithNoData = np.where(np.isnan(array), -9, array)
    pretext = """ncols         %s
nrows         %s
xllcorner     -180
yllcorner     -90
cellsize      %s
NODATA_value  -9
""" % (
        str(ncols),
        str(nrows),
        str(cellsizelats),
    )
    file1.write(pretext)
    print(len(arrayWithNoData))
    print(min(arrayWithNoData))
    print(max(arrayWithNoData))
    flippedarr = np.ravel(np.flipud(arrayWithNoData.reshape((ncols, nrows))))
    file1.write(" ".join(map(str, flippedarr)))
    file1.close()


# create a global ascii at 5 minute resolution
def create5minASCII(df, column, fn):
    file1 = open(fn + ".asc", "w")  # write mode
    array = np.array(df[column].values).astype("float32")
    arrayWithNoData = np.where(np.bitwise_or(array < 0, np.isnan(array)), -9, array)
    pretext = """ncols         4320
    nrows         2160
    xllcorner     -180
    yllcorner     -90
    cellsize      0.083333333333333
    NODATA_value  -9
    """
    file1.write(pretext)
    print(len(arrayWithNoData))
    print(min(arrayWithNoData))
    print(max(arrayWithNoData))
    flippedarr = np.ravel(
        np.flipud(np.transpose(arrayWithNoData.reshape((4320, 2160))))
    )
    file1.write(" ".join(map(str, flippedarr)))
    file1.close()


# create a global ascii at 5 minute resolution including negative alues
def create5minASCIIneg(df, column, fn):
    file1 = open(fn + ".asc", "w")  # write mode
    array = np.array(df[column].values).astype("float32")
    arrayWithNoData = np.where(np.isnan(array), -9, array)
    pretext = """ncols         4320
nrows         2160
xllcorner     -180
yllcorner     -90
cellsize      0.083333333333333
NODATA_value  -9
"""
    file1.write(pretext)
    print(len(arrayWithNoData))
    print(min(arrayWithNoData))
    print(max(arrayWithNoData))
    flippedarr = np.ravel(
        np.flipud(np.transpose(arrayWithNoData.reshape((4320, 2160))))
    )
    file1.write(" ".join(map(str, flippedarr)))
    file1.close()
