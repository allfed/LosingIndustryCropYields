'''''
This code imports and downsamples a raster (geotiff) of livestock area
Uses DA (Dasymetric) part of dataset.



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
livestocks = ['Bf','Dk','Gt','Pg','Sh','Ho','Ct','Ch']

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


print('reading livestock count data')
for l in livestocks:

	#total animals per pixel
	ldata=rasterio.open(params.livestockDataLoc+'5_'+l+'_2010_Da.tif')
	lArr=ldata.read(1)

	latbins=np.floor(len(lArr)/len(lats)).astype('int')
	lonbins=np.floor(len(lArr[0])/len(lons)).astype('int')
	lArrResized=lArr[0:latbins*len(lats),0:lonbins*len(lons)]
	print('done reading '+l)

	#make data zero if data < 0.
	lArrResizedZeroed=np.where(lArrResized<0, 0, lArrResized)

	lBinned= utilities.rebinCumulative(lArrResizedZeroed, sizeArray)
	lBinnedReoriented=np.fliplr(np.transpose(lBinned))

	data[l] = pd.Series(lBinnedReoriented.ravel())

df = pd.DataFrame(data=data)
geometry = gpd.points_from_xy(df.lons, df.lats)
gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
grid= utilities.makeGrid(gdf)
grid.to_pickle(params.geopandasDataDir + "Livestock.pkl")

title="Chickens, 2010"
label="Heads chickens in ~2 each degree square cell"
Plotter.plotMap(grid,'Ch',title,label,'HeadsCattle',True)

totalHeadsOfChickens=grid['Ch'].sum()
print("Total Heads Of Chickens: "+str(totalHeadsOfChickens))

title="Cattle, 2010"
label="Heads cattle in ~2 each degree square cell"
Plotter.plotMap(grid,'Ct',title,label,'HeadsCattle',True)

totalHeadsOfCattle=grid['Ct'].sum()
print("Total Heads Of Cattle: "+str(totalHeadsOfCattle))
