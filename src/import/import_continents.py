# -*- coding: utf-8 -*-
"""
This code imports a raster file containing information about the borders of the continents.

The continents are coded as followed:
    0 = No Data
    1 = Africa
    2 = Asia
    3 = Oceania
    4 = North America
    5 = Europe
    6 = South America
    7 = Antarctica

Created on Sat Apr 30 22:35:33 2022

@author: Jessica Mörsdorf
jessica@allfed.info
"""
import src.utilities.utilities as utilities
import src.utilities.params as params  # get file location and varname parameters for data import
import src.utilities.plotter as Plotter


import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import os

# load the params from the params.ods file into the params object
params.importIfNotAlready()

MAKE_GRID = False

cdata = rasterio.open(params.continentDataLoc + "/continents_raster_5arcmin.tif")

print("reading continent data")
cArr = cdata.read(1)
print("done reading")

lats = np.linspace(
    -90, 90 - params.latdiff, np.floor(180 / params.latdiff).astype("int")
)
lons = np.linspace(
    -180, 180 - params.londiff, np.floor(360 / params.londiff).astype("int")
)

latbins = np.floor(len(cArr) / len(lats)).astype("int")
lonbins = np.floor(len(cArr[0]) / len(lons)).astype("int")

# I'm relatively sure that these steps are only necessary if the file is supposed to be downsampled
cArrResized = cArr[0 : latbins * len(lats), 0 : lonbins * len(lons)]
sizeArray = [len(lats), len(lons)]

# substitute all negative values (nodata values) with 0
# not necessary in this case, as no data is already coded as 0
cArrResizedFiltered = np.where(cArrResized < 0, 0, cArrResized)
# substitute values 8 and 9 as there are only 7 continents
cArrResizedFiltered = np.where(cArrResizedFiltered == 8, 5, cArrResizedFiltered)
cArrResizedFiltered = np.where(cArrResizedFiltered == 9, 3, cArrResizedFiltered)

# I'm relatively sure that this step are only necessary if the file is supposed to be downsampled
cBinned = utilities.rebinCumulative(cArrResizedFiltered, sizeArray)
# not exactly sure why this is done: switches everything up but no idea why
cBinnedReoriented = np.fliplr(np.transpose(cBinned))

lats2d, lons2d = np.meshgrid(lats, lons)

cdata = {
    "lats": pd.Series(lats2d.ravel()),
    "lons": pd.Series(lons2d.ravel()),
    "continent": pd.Series(cBinnedReoriented.ravel()),
}

cdf = pd.DataFrame(data=cdata)
cdf["continent"] = cdf["continent"].astype("int8")
if MAKE_GRID:
    geometry = gpd.points_from_xy(cdf.lons, cdf.lats)
    gcdf = gpd.GeoDataFrame(cdf, crs={"init": "epsg:4326"}, geometry=geometry)
    grid = utilities.makeGrid(gcdf)
    grid.to_csv(params.geopandasDataDir + "CropYield.csv")

    plotGrowArea = True

    title = "Continents "
    label = "Grow Area (ha)"
    Plotter.plotMap(grid, "growArea", title, label, "CropGrowArea", plotGrowArea)

else:
    assert cdf["lats"].iloc[-1] > cdf["lats"].iloc[0]
    assert cdf["lons"].iloc[-1] > cdf["lons"].iloc[0]

    os.makedirs(params.inputDataDir, exist_ok=True)
    cdf.to_pickle(params.inputDataDir + "Continents.pkl", compression="zip")

# create ASCII to ensure that the process worked the way it is intended
#os.makedirs(params.inputAsciiDir, exist_ok=True)
#utilities.create5minASCIIneg(cdf, 'continent', params.inputAsciiDir+'Continents')
