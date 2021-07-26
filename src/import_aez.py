'''''
This code imports and downsamples a raster (geotiff) of agroecological zones 
(AEZ). 


Morgan Rivers
morgan@allfed.info
7/24/21
'''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from src import params  # get file location and varname parameters for data import
from src.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import utilities

#load the params from the params.ods file into the params object
params.importIfNotAlready()

# aquastat=pd.read_csv(params.aquastatIrrigationDataLoc,index_col=False)
# aq=aquastat.dropna(how='all').replace('',np.nan)
AEZs = [
'thz_class_CRUTS32_Hist_8110_100_avg.tif',
'mst_class_CRUTS32_Hist_8110_100_avg.tif',
'soil_regime_CRUTS32_Hist_8110.tif'
]

# we ignore the last latitude cell, and generate what the eventual (lowres)
# grid will look like
lats = np.linspace(-90, 90 - params.latdiff, \
			   np.floor(180 / params.latdiff).astype('int'))
lons = np.linspace(-180, 180 - params.londiff, \
			   np.floor(360 / params.londiff).astype('int'))

lats2d, lons2d = np.meshgrid(lats, lons)

sizeArray=[len(lats),len(lons)]

data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel())}


print('reading agroecological zone data')
for z in AEZs:
	#total animals per pixel
	zdata=rasterio.open(params.aezDataLoc+z)
	zname=z.split('_')[0]
	zArr=zdata.read(1)

	latbins=np.floor(len(zArr)/len(lats)).astype('int')
	lonbins=np.floor(len(zArr[0])/len(lons)).astype('int')
	zArrResized=zArr[0:latbins*len(lats),0:lonbins*len(lons)]
	print('done reading '+z)

	#make data zero if data < 0.
	zArrResizedZeroed=np.where(zArrResized<0, 0, zArrResized)

	zBinned= utilities.rebinCumulative(zArrResizedZeroed, sizeArray)
	zBinnedReoriented=np.fliplr(np.transpose(zBinned))

	data[zname] = pd.Series(zBinnedReoriented.ravel())

df = pd.DataFrame(data=data)
geometry = gpd.points_from_xy(df.lons, df.lats)
gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
grid= utilities.makeGrid(gdf)
grid.to_pickle(params.geopandasDataDir + "AEZ.pkl")

title="AEZ"
label="AEZ in ~2 each degree square cell"
Plotter.plotMap(grid,'mst',title,label,'mstZone',True)

# totalHeadsOfChickens=grid['Ch'].sum()
# print("Total Heads Of Chickens: "+str(totalHeadsOfChickens))

# title="Cattle, 2010"
# label="Heads cattle in ~2 each degree square cell"
# Plotter.plotMap(grid,'Ct',title,label,'HeadsCattle',True)

# totalHeadsOfCattle=grid['Ct'].sum()
# print("Total Heads Of Cattle: "+str(totalHeadsOfCattle))
