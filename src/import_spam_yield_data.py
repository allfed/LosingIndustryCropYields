'''''
This code imports a raster (geotiff) of crop yield and area by crop from 
SPAM2010

yield units are given as kg per hectare for yield, and hectare for area.

see
https://essd.copernicus.org/articles/12/3545/2020/essd-12-3545-2020.pdf
"We prepare datafor the model based on the 2009–2011 average of the cropproduction  statistics"

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
from src import utilities

#load the params from the params.ods file into the params object
params.importIfNotAlready()

for crop in params.allCrops:

	ydata=rasterio.open(params.spamCropYieldDataLoc+'spam2010V2r0_global_Y_'+crop+'_A.tif')
	adata=rasterio.open(params.spamCropYieldDataLoc+'spam2010V2r0_global_H_'+crop+'_A.tif')

	print('reading grow area and yield data')
	yArr=ydata.read(1)
	aArr=adata.read(1)
	print('done reading')

	lats = np.linspace(-90, 90 - params.latdiff, \
				   np.floor(180 / params.latdiff).astype('int'))
	lons = np.linspace(-180, 180 - params.londiff, \
				   np.floor(360 / params.londiff).astype('int'))

	latbins=np.floor(len(yArr)/len(lats)).astype('int')
	lonbins=np.floor(len(yArr[0])/len(lons)).astype('int')

	yArrResized=yArr[0:latbins*len(lats),0:lonbins*len(lons)]
	aArrResized=aArr[0:latbins*len(lats),0:lonbins*len(lons)]
	sizeArray=[len(lats),len(lons)]

	yArrResizedFiltered=np.where(yArrResized<0, 0, yArrResized)
	aArrResizedFiltered=np.where(aArrResized<0, 0, aArrResized)
	tArrResizedFiltered=np.multiply(yArrResizedFiltered,aArrResizedFiltered)

	aBinned= utilities.rebinCumulative(aArrResizedFiltered, sizeArray)
	aBinnedReoriented=np.fliplr(np.transpose(aBinned))
	tBinned = utilities.rebinCumulative(tArrResizedFiltered, sizeArray)
	tBinnedReoriented=np.fliplr(np.transpose(tBinned))

	lats2d, lons2d = np.meshgrid(lats, lons)

	data = {"lats": pd.Series(lats2d.ravel()),
			"lons": pd.Series(lons2d.ravel()),
			"growArea": pd.Series(aBinnedReoriented.ravel()),
			"totalYield": pd.Series(tBinnedReoriented.ravel())
			}
	data['yield_kgPerHa'] = (data['totalYield'] / data['growArea'])

	df = pd.DataFrame(data=data)
	geometry = gpd.points_from_xy(df.lons, df.lats)
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)

	grid= utilities.makeGrid(gdf)

	grid.to_pickle(params.geopandasDataDir + crop + "CropYield.pkl")

	plotGrowArea=True

	title="Average Global Grow Area "+ crop + " for Years 2009-2011"
	label="Grow Area (ha)"
	Plotter.plotMap(grid,'growArea',title,label,'CropGrowArea',plotGrowArea)

	title="Average Global Yield "+ crop + " for Years 2009-2011"
	label="Yield (kg/ha)"
	Plotter.plotMap(grid,'yield_kgPerHa',title,label,'CropYield',plotGrowArea)

	print("data['totalYield'].sum()/data['growArea'].sum()")
	print(data['totalYield'].sum()/data['growArea'].sum())
	print("data['yield_kgPerHa'].mean()")
	print((data['yield_kgPerHa']*data['growArea']).sum()/data['growArea'].sum())

	print("total yield, tonnes, "+crop+": "+str(grid['totalYield'].sum()/1000))
