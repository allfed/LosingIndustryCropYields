"""''
This code imports and downsamples a raster (ascii) of fertilizer application 
rates from the pangea dataset. https://doi.pangaea.de/10.1594/PANGAEA.863323

imported units are g N/m**2, but we find kg/m**2 by dividing by 1000. 

Because the original is half degree and all our other datasets are 5 minute,
we upsample the array.

Note: the raw data last 6 degrees of longitude are missing! 

Morgan Rivers
morgan@allfed.info
7/24/21
"""
import src.utilities.utilities as utilities
import src.utilities.params as params  # get file location and varname parameters for data import
import src.utilities.plotter as Plotter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from geopandas.tools import sjoin
import os

# import datetime

# import gc
# import os, psutil
# from pympler import summary
# from pympler import muppy

# process = psutil.Process(os.getpid())
# print("mem1: "+str(process.memory_info().rss/1e6))  # in megabytes
# from pympler.tracker import SummaryTracker
# tracker = SummaryTracker()

# load the params from the params.ods file into the params object
params.importIfNotAlready()
# params.deleteGlobals()

MAKE_GRID = False

mn_lat = -88.5
mx_lat = 88.5
mn_lon = -180
mx_lon = 180

# 5 arcminutes in degrees
five_minute = 5 / 60


pSums = {}
nbins = params.growAreaBins


start_lat_index = np.floor((90 - mx_lat) / five_minute).astype("int")
start_lon_index = np.floor((mn_lon + 180) / five_minute).astype("int")

# we ignore the last latitude cell
lats = np.linspace(
    -90, 90 - params.latdiff, np.floor(180 / params.latdiff).astype("int")
)
lons = np.linspace(
    -180, 180 - params.londiff, np.floor(360 / params.londiff).astype("int")
)

result = np.full((nbins * len(lats), nbins * len(lons)), np.nan)

lats2d, lons2d = np.meshgrid(lats, lons)
data = {"lats": pd.Series(lats2d.ravel()), "lons": pd.Series(lons2d.ravel())}
df = pd.DataFrame(data=data)

sizeArray = [len(lats), len(lons)]


fertilizers = ["n", "p"]


print("reading fertilizer data")
for f in fertilizers:
    # start_time = datetime.datetime.now()

    fdata = rasterio.open(params.fertilizerDataLoc + f + "fery2013.asc")
    fArr = fdata.read(1)

    # so, 1/2 degree= 30 arcminutes=6 by 5 arcminute chunks
    # also, convert grams to kg.
    fArrUpsampled = fArr.repeat(6, axis=0).repeat(6, axis=1) / 1000
    result[
        start_lat_index : start_lat_index + len(fArrUpsampled),
        start_lon_index : start_lon_index + len(fArrUpsampled[0]),
    ] = fArrUpsampled

    if MAKE_GRID:
        fArrResized = result[0 : nbins * len(lats), 0 : nbins * len(lons)]
        fBinned = utilities.rebin(fArrResized, sizeArray)
        fBinnedReoriented = np.fliplr(np.transpose(fBinned))
        df[f] = pd.Series(fBinnedReoriented.ravel())
    else:
        df[f] = pd.Series(np.fliplr(np.transpose(result)).ravel())

    print("done reading " + f)

if MAKE_GRID:
    geometry = gpd.points_from_xy(df.lons, df.lats)
    gdf = gpd.GeoDataFrame(df, crs={"init": "epsg:4326"}, geometry=geometry)
    grid = utilities.makeGrid(gdf)

    title = "Nitrogen Fertilizer Application, no manure"
    label = "kg/m^2/year "
    Plotter.plotMap(grid, "n", title, label, "NitrogenFertilizer", True)

    title = "Phosphorus Fertilizer Application, no manure"
    label = "kg/m^2/year "
    Plotter.plotMap(grid, "p", title, label, "PhosphorusFertilizer", True)

    grid.to_csv(params.geopandasDataDir + "Fertilizer.csv")
else:
    assert df["lats"].iloc[-1] > df["lats"].iloc[0]
    assert df["lons"].iloc[-1] > df["lons"].iloc[0]
    os.makedirs(params.inputDataDir, exist_ok=True)
    df.to_pickle(params.inputDataDir + "FertilizerHighRes.pkl", compression="zip")
