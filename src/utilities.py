'''

Useful functions that don't involve plotting, called from various locations in the code.

'''

from src import params

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely

params.importIfNotAlready()

#this function is used to make a bunch of rectangle grid shapes so the 
#plotting looks nice and so we can later add up the crop area inside the grid
def makeGrid(df):
	xmin=np.min(df['lats'])
	ymin=np.min(df['lons'])
	xmax=np.max(df['lats'])
	ymax=np.max(df['lons'])
	cells=[]
	for index,row in df.iterrows():
		cell=shapely.geometry.box(row['lons'], row['lats'], row['lons'] + params.londiff, row['lats'] + params.latdiff)
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
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:3857'}, geometry=geometry)

	grid = makeGrid(gdf)
	fn = params.geopandasDataDir + name + '.pkl'

	grid=grid.sort_values(by=['lats', 'lons'])
	grid.to_pickle(fn)


#save a .pkl file with the gridded data saved in columns labelled by month 
#number
def saveDictasgeopandas(name,data):
	print(len(data))

	df = pd.DataFrame(data)
	geometry = gpd.points_from_xy(df.lons, df.lats)
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:3857'}, geometry=geometry)
	grid=makeGrid(gdf)
	fn= "../" + params.geopandasDataDir + name + '.pkl'

	grid=grid.sort_values(by=['lats', 'lons'])
	grid.to_pickle(fn)

	return grid