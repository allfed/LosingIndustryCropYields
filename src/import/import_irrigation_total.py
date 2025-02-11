"""''
https://aquaknow.jrc.ec.europa.eu/en/content/global-map-irrigated-areas-v50-gmia

download: http://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version


This code imports a raster (geotiff) of crop irrigation area
 from gmiav5 at 5 minute resolution.

saves it as a csv with ~900 million rows 

see
https://essd.copernicus.org/articles/12/3545/2020/essd-12-3545-2020.pdf
"We prepare datafor the model based on the 2009–2011 average of the cropproduction  statistics"

Output of Import: all units are per cell in hectares
    'area' column: total irrigated area 

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
import os
from geopandas.tools import sjoin

# load the params from the params.ods file into the params object
params.importIfNotAlready()

five_minute = 5 / 60
# total area
fracEquippedData = rasterio.open(params.irrigationDataLoc + "gmia_v5_aei_pct.asc")
fracActuallyIrrigatedData = rasterio.open(
    params.irrigationDataLoc + "gmia_v5_aai_pct_aei.asc"
)

print("reading irrigation fraction data")
eArr = fracEquippedData.read(1)
aiArr = fracActuallyIrrigatedData.read(1)  # fracEquippedData was used twice
fArr = np.multiply(eArr, aiArr) / 100 / 100
# we ignore the last latitude cell
lats = np.linspace(-90, 90 - five_minute, np.floor(180 / five_minute).astype("int"))
lons = np.linspace(-180, 180 - five_minute, np.floor(360 / five_minute).astype("int"))

latbins = np.floor(len(fArr) / len(lats)).astype("int")
lonbins = np.floor(len(fArr[0]) / len(lons)).astype("int")

fArrResized = fArr[0 : latbins * len(lats), 0 : lonbins * len(lons)]

# areaBinned= utilities.rebinCumulative(areaArrResized, sizeArray)
# areaBinnedReoriented=np.fliplr(np.transpose(areaBinned))
# swBinned= utilities.rebinCumulative(swArrResizedFiltered, sizeArray)
# swBinnedReoriented=np.fliplr(np.transpose(swBinned))
# gwBinned = utilities.rebinCumulative(gwArrResizedFiltered, sizeArray)
# gwBinnedReoriented=np.fliplr(np.transpose(gwBinned))

lats2d, lons2d = np.meshgrid(lats, lons)
data = {
    "lats": pd.Series(lats2d.ravel()),
    "lons": pd.Series(lons2d.ravel()),
    "fraction": pd.Series(np.fliplr(np.transpose(fArrResized)).ravel()),
}

df = pd.DataFrame(data=data)

# fraction of cell that's irrigated
os.makedirs(params.inputDataDir, exist_ok=True)
df.to_pickle(params.inputDataDir + "FracIrrigationAreaHighRes.pkl", compression="zip")
