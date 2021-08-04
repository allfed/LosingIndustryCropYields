'''

Useful functions that don't involve plotting, called from various locations in the code.

'''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from src import params

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely

params.importIfNotAlready()

#this function is used to make a bunch of rectangle grid shapes so the 
#plotting looks nice and so we can later add up the crop area inside the grid
def makeGrid(df):
	nbins=params.growAreaBins
	cell_size_lats=params.latdiff
	cell_size_lons=params.londiff
	# mn_lat=-56.00083
	# mx_lat=83.99917
	# mn_lon=-178.875
	# mx_lon=179.875
	# raw_lats = np.linspace(mn_lat, mx_lat,  1681)
	# raw_lons = np.linspace(mn_lon, mx_lon,  4306)
	# cell_size_lats=(raw_lats[1]-raw_lats[0])*(nbins)
	# cell_size_lons=(raw_lons[1]-raw_lons[0])*(nbins)


	cells=[]
	for index,row in df.iterrows():
		cell=shapely.geometry.box(row['lons'], row['lats'], row['lons'] + params.londiff, row['lats'] + cell_size_lats)#params.latdiff)
		cells.append(cell)
	crs={'init':'epsg:4326'}
	geo_df=gpd.GeoDataFrame(df,crs=crs,geometry=cells)
	geo_df=geo_df.sort_values(by=['lats', 'lons'])
	geo_df = geo_df.reset_index(drop=True)
	return geo_df

# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
# downsample the 2d array so that crop percentages are averaged.
def rebin(a, shape):
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	return a.reshape(sh).mean(-1).mean(1)

# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
# downsample the 2d array so that crop percentages are averaged.
def rebinIgnoreZeros(a, shape):
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	asmall = a.reshape(sh).mean(-1).mean(1)
	
	#when zero, true, false otherwise. Number gives fraction of cells that are
	#nonzero
	anonzeros = 1-(a==0).reshape(sh).mean(-1).mean(1)

	#if all cells are zero, report nan for yield
	anonzeronan = np.where(anonzeros==0,np.nan,anonzeros)
	
	#multiply average by fraction zero cells, to cancel out their effect
	#cell_avg=cell_sum/ncells => 25% zero cells would be
	#cell_avg=cell_sum_nonzeros/(cells_zero+cells_nonzero)
	#cell_avg_nonzero=cell_sum_nonzeros/(cells_nonzero)
	#therefore
	#cell_avg_nonzero=cell_avg*(cells_zero+cells_nonzero)/(cells_nonzero)
	#cell_avg_nonzero=cell_avg/(fraction_cells_nonzero)

	return np.divide(asmall,anonzeronan)


# https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
# downsample the 2d array, but add all the values together.
def rebinCumulative(a, shape):
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	asmall = a.reshape(sh).mean(-1).mean(1)
	product=np.product((np.array(a.shape)/np.array(shape)))
	return asmall*product

#save a .pkl file with the gridded data saved in columns labelled by month 
#number
#(save the imported nuclear winter data in geopandas format)
def saveasgeopandas(name,allMonths,gridAllMonths,lats,lons):

	assert(len(allMonths)==len(gridAllMonths))

	# create 2D arrays from 1d latitude, longitude arrays
	lats2d, lons2d = np.meshgrid(lats, lons)

	data = {"lats": pd.Series(lats2d.ravel()),
			"lons": pd.Series(lons2d.ravel())}
	
	
	for i in range(0,len(allMonths)):
		grid=gridAllMonths[i]
		month=allMonths[i]
		data[month]=pd.Series(grid.ravel())
	df = pd.DataFrame(data=data)
	geometry = gpd.points_from_xy(df.lons, df.lats)
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)

	grid = makeGrid(gdf)
	fn = params.geopandasDataDir + name + '.csv'

	grid=grid.sort_values(by=['lats', 'lons'])
	return grid
	# grid.to_csv(fn)


#save a .pkl file with the gridded data saved in columns labelled by month 
#number
def saveDictasgeopandas(name,data):
	df = pd.DataFrame(data)
	geometry = gpd.points_from_xy(df.lons, df.lats)
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
	grid=makeGrid(gdf)
	fn= params.geopandasDataDir + name + '.pkl'

	grid=grid.sort_values(by=['lats', 'lons'])
	grid.to_pickle(fn)

	return grid
