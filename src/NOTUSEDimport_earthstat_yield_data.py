'''''
NOT CURRENTLY USED

This code imports a raster (geotiff) of crop yield and area by crop from 
earthstat. Imported data is averaged from 2000-2005. 

A full description can be found here:
www.earthstat.org/harvested-area-yield-4-crops-1995-2005/

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

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import utilities

#load the params from the params.ods file into the params object
params.importIfNotAlready()

for crop in params.allCrops:

	ydata=rasterio.open(params.cropYieldDataLoc+crop+'_2005_Yield.tif')
	adata=rasterio.open(params.cropYieldDataLoc+crop+'_2005_Area.tif')

	print('reading grow area and yield')
	yArr=ydata.read(1)
	aArr=adata.read(1)
	print('done reading')

	# we ignore the last latitude cell
	lats = np.linspace(-90, 90 - params.latdiff, \
				   np.floor(180 / params.latdiff).astype('int'))
	lons = np.linspace(-180, 180 - params.londiff, \
				   np.floor(360 / params.londiff).astype('int'))

	latbins=np.floor(len(yArr)/len(lats)).astype('int')
	lonbins=np.floor(len(yArr[0])/len(lons)).astype('int')

	yArrResized=yArr[0:latbins*len(lats),0:lonbins*len(lons)]
	aArrResized=aArr[0:latbins*len(lats),0:lonbins*len(lons)]
	sizeArray=[len(lats),len(lons)]
	yArrResizedFiltered=1000/np.where(yArrResized<=0, 1000000000, yArrResized)
	aArrResizedFiltered=np.where(aArrResized<0, 0, aArrResized)
	tArrResizedFiltered=np.multiply(aArrResizedFiltered,yArrResizedFiltered)
	tBinned= utilities.rebin(tArrResizedFiltered, sizeArray)
	tBinnedReoriented=np.fliplr(np.transpose(tBinned))

	lats2d, lons2d = np.meshgrid(lats, lons)

	data = {"lats": pd.Series(lats2d.ravel()),
			"lons": pd.Series(lons2d.ravel()),
			# crop area 1000s of km (hectares).
			"growArea": pd.Series(tBinnedReoriented.ravel())}

	print(data)
	df = pd.DataFrame(data=data)
	geometry = gpd.points_from_xy(df.lons, df.lats)
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)

	grid= utilities.makeGrid(gdf)

	grid.to_pickle(params.geopandasDataDir + crop + "CropYield.pkl")

	plotGrowArea=True

	label="Yield (tons per hectare) Crop Growing, "+ crop
	title="Average Crop Tons/Ha for Years 2003-2007"
	Plotter.plotMap(grid,'growArea',title,label,'CropGrowFraction',plotGrowArea)