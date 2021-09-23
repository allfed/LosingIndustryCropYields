'''''
This code imports and downsamples a raster (ascii) of fertilizer application 
rates from the pangea dataset. https://doi.pangaea.de/10.1594/PANGAEA.863323

imported units are g N/m**2, but we find kg/m**2 by dividing by 1000. 

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

# import datetime

# import gc
# import os, psutil
# from pympler import summary
# from pympler import muppy

import resource
from sys import platform
if platform == "linux" or platform == "linux2":
	#this is to ensure Morgan's computer doesn't crash
	import resource
	rsrc = resource.RLIMIT_AS
	resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb

# process = psutil.Process(os.getpid())
# print("mem1: "+str(process.memory_info().rss/1e6))  # in megabytes 
# from pympler.tracker import SummaryTracker
# tracker = SummaryTracker()

#load the params from the params.ods file into the params object
params.importIfNotAlready()
# params.deleteGlobals()

MAKE_GRID = False

mn_lat=-88.5
mx_lat=88.5
mn_lon=-180
mx_lon=180

#5 arcminutes in degrees
five_minute=5/60

# raw_lats = np.linspace(mn_lat, mx_lat,  np.floor((mx_lat-mn_lat)/five_minute).astype('int')) #5 minute res
# raw_lons = np.linspace(mn_lon, mx_lon,  np.floor((mx_lon-mn_lon)/five_minute).astype('int')) #5 minute res

# print(raw_lats)
# print(raw_lons)
# print(len(raw_lats))
# print(len(raw_lons))

pSums={}
nbins=params.growAreaBins


start_lat_index=np.floor((90-mx_lat)/five_minute).astype('int')
start_lon_index=np.floor((mn_lon+180)/five_minute).astype('int')

# we ignore the last latitude cell
lats = np.linspace(-90, 90 - params.latdiff, \
				   np.floor(180 / params.latdiff).astype('int'))
lons = np.linspace(-180, 180 - params.londiff, \
				   np.floor(360 / params.londiff).astype('int'))

result=np.full((nbins*len(lats),nbins*len(lons)),np.nan)

lats2d, lons2d = np.meshgrid(lats, lons)
data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel())}
df = pd.DataFrame(data=data)

sizeArray=[len(lats),len(lons)]


fertilizers = ['n','p']


print('reading fertilizer data')
for f in fertilizers:

	# start_time = datetime.datetime.now()

	fdata=rasterio.open(params.fertilizerDataLoc+f+'fery2013.asc')
	fArr=fdata.read(1)

	# so, 1/2 degree= 30 arcminutes=6 by 5 arcminute chunks
	# also, convert grams to kg.
	print(len(fArr))
	print(len(fArr[0]))
	print(fArr[100:110][9])
	fArrUpsampled=fArr.repeat(6, axis=0).repeat(6, axis=1)/1000
	result[start_lat_index:start_lat_index+len(fArrUpsampled),start_lon_index:start_lon_index+len(fArrUpsampled[0])]=fArrUpsampled
	
	if(MAKE_GRID):
		# quit()


		# time1 = datetime.datetime.now()

		fArrResized=result[0:nbins*len(lats),0:nbins*len(lons)]
		# time2 = datetime.datetime.now()
		# fArrResizedZeroed=np.where(fArrResized<0, 0, fArrResized)
		# time3 = datetime.datetime.now()
		fBinned= utilities.rebin(fArrResized, sizeArray)
		fBinnedReoriented=np.fliplr(np.transpose(fBinned))
		df[f]=pd.Series(fBinnedReoriented.ravel())
	else:
		df[f]=pd.Series(np.fliplr(np.transpose(result)).ravel())
		# fBinnedReoriented=np.fliplr(np.transpose(fArrUpsampled))

		# strout=''
		# for a in fBinnedReoriented[1000:1100][90]:
		# 	strout = strout + ' ' + str(a)
		# print(strout)
	# time4 = datetime.datetime.now()
	# time5 = datetime.datetime.now()


	# time6 = datetime.datetime.now()



	print('done reading '+f)

# df = pd.DataFrame(data=data)
# time7 = datetime.datetime.now()
if(MAKE_GRID):

	geometry = gpd.points_from_xy(df.lons, df.lats)
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
	grid= utilities.makeGrid(gdf)

	# title="Nitrogen Fertilizer Application, no manure"
	# label="kg/m^2/year "
	# Plotter.plotMap(grid,'n',title,label,'NitrogenFertilizer',True)

	# title="Phosphorus Fertilizer Application, no manure"
	# label="kg/m^2/year "
	# Plotter.plotMap(grid,'p',title,label,'PhosphorusFertilizer',True)

	grid.to_csv(params.geopandasDataDir + "Fertilizer.csv")
else:
	assert(df['lats'].iloc[-1]>df['lats'].iloc[0])
	assert(df['lons'].iloc[-1]>df['lons'].iloc[0])
	# quit()
	# df.sort_values(by=['lats', 'lons'],inplace=True)
	# print('2')
	# df = df.reset_index(drop=True)
	df.to_csv(params.geopandasDataDir + "FertilizerHighRes.csv")
	# time10 = datetime.datetime.now()


# print('time1: '+str((time1-start_time).total_seconds() * 1000))
# print('time2: '+str((time2-time1).total_seconds() * 1000))
# print('time3: '+str((time3-time2).total_seconds() * 1000))
# print('time4: '+str((time4-time3).total_seconds() * 1000))
# print('time5: '+str((time5-time4).total_seconds() * 1000))
# print('time6: '+str((time6-time5).total_seconds() * 1000))
# print('time7: '+str((time7-time6).total_seconds() * 1000))
# print('time8: '+str((time8-time7).total_seconds() * 1000))
# print('time9: '+str((time9-time8).total_seconds() * 1000))
# print('time10: '+str((time10-time9).total_seconds() * 1000))
# params.deleteGlobals()

print('5')
# all_objects = muppy.get_objects()
# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)
# print("mem2: "+str(psutil.Process().memory_info().rss/1e6))  # in megabytes 
# gc.collect()
# print("mem3: "+str(psutil.Process().memory_info().rss/1e6))  # in megabytes 
