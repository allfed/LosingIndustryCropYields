'''''
This code imports and downsamples a raster (ascii) of fertilizer application 
rates from the pangea dataset. https://doi.pangaea.de/10.1594/PANGAEA.863323

Because the original is half degree and all our other datasets are 5 minute,
we upsample the array.

Note: the raw data last 6 degrees of longitude are missing! 

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
from geopandas.tools import sjoin
import rasterio
import utilities

#load the params from the params.ods file into the params object
params.importIfNotAlready()


mn_lat=-88.5
mx_lat=88.5
mn_lon=-180
mx_lon=180

#5 arcminutes in degrees
five_minute=5/60

raw_lats = np.linspace(mn_lat, mx_lat,  np.floor((mx_lat-mn_lat)/five_minute).astype('int')) #5 minute res
raw_lons = np.linspace(mn_lon, mx_lon,  np.floor((mx_lon-mn_lon)/five_minute).astype('int')) #5 minute res

print(raw_lats)
print(raw_lons)
print(len(raw_lats))
print(len(raw_lons))

pSums={}
nbins=params.growAreaBins


start_lat_index=np.floor((90-mx_lat)/five_minute).astype('int')
start_lon_index=np.floor((mn_lon+180)/five_minute).astype('int')

# we ignore the last latitude cell
lats = np.linspace(-90, 90 - params.latdiff, \
				   np.floor(180 / params.latdiff).astype('int'))
lons = np.linspace(-180, 180 - params.londiff, \
				   np.floor(360 / params.londiff).astype('int'))

result=np.zeros((nbins*len(lats),nbins*len(lons)))

lats2d, lons2d = np.meshgrid(lats, lons)
data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel())}
# df = pd.DataFrame(data=data)

#make geometry
# geometry = gpd.points_from_xy(df.lons, df.lats)
# gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
# grid= utilities.makeGrid(gdf)

sizeArray=[len(lats),len(lons)]




fertilizers = ['n','p']


print('reading fertilizer data')
for f in fertilizers:


	fdata=rasterio.open(params.fertilizerDataLoc+f+'fery2013.asc')
	fArr=fdata.read(1)

	# so, 1/2 degree= 30 arcminutes=6 by 5 arcminute chunks
	# also, convert grams to kg.
	fArrUpsampled=fArr.repeat(6, axis=0).repeat(6, axis=1)/1000

	result[start_lat_index:start_lat_index+len(fArrUpsampled),start_lon_index:start_lon_index+len(fArrUpsampled[0])]=fArrUpsampled


	fArrResized=result[0:nbins*len(lats),0:nbins*len(lons)]
	fArrResizedZeroed=np.where(fArrResized<0, 0, fArrResized)
	fBinned= utilities.rebin(fArrResizedZeroed, sizeArray)
	fBinnedReoriented=np.fliplr(np.transpose(fBinned))

	data[f]=pd.Series(fBinnedReoriented.ravel())



	print('done reading '+f)

df = pd.DataFrame(data=data)
geometry = gpd.points_from_xy(df.lons, df.lats)
gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4269'}, geometry=geometry)

grid= utilities.makeGrid(gdf)
grid.to_pickle(params.geopandasDataDir + "Fertilizer.pkl")

title="Nitrogen Fertilizer Application, "
label="kg N/m^2/year "
Plotter.plotMap(grid,'n',title,label,'NitrogenFertilizer',True)


