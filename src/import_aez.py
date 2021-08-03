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
'mst_class_CRUTS32_Hist_8110_100_avg.tif',
'thz_class_CRUTS32_Hist_8110_100_avg.tif'
# 'soil_regime_CRUTS32_Hist_8110.tif'
]

mn_lat=-90
mx_lat=90
mn_lon=-180
mx_lon=180

nbins=params.growAreaBins

#5 arcminutes in degrees
five_minute=5/60

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
df = pd.DataFrame(data=data)

#make geometry
geometry = gpd.points_from_xy(df.lons, df.lats)
gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
grid= utilities.makeGrid(gdf)

sizeArray=[len(lats),len(lons)]


#it's a bit wierd to pull from another dataset, but the livestock dataset
# conveniently had the area of each 5 minute raster cell available
#below we use the area of each cell to estimate the relative weight of the zone 
#when comparing the most commmon zone in a region
adata=rasterio.open(params.livestockDataLoc+'8_Areakm.tif')
aArr=adata.read(1)

zonetypes={}
zonetypes['thz']=[0,1,2,3,4,5,6,7,8,9,10]
zonetypes['mst']=[0,1,2,3,4,5,6,7]
print('reading agroecological zone data')
df_tmp = pd.DataFrame(data=data)

for z in AEZs:
	zname=z.split('_')[0]
	zdata=rasterio.open(params.aezDataLoc+z)
	zArr=zdata.read(1)
	# print(zArr[1500:1550,1400:1420])
	for zt in zonetypes[zname]:
		areas = (zArr==zt)

		latbins=np.floor(len(areas)/len(lats)).astype('int')
		lonbins=np.floor(len(areas[0])/len(lons)).astype('int')
		zArrResized=areas[0:latbins*len(lats),0:lonbins*len(lons)]
		print('done reading '+z)

		grid_area=np.multiply(zArrResized,aArr)

		zBinned= utilities.rebinCumulative(grid_area, sizeArray)
		zBinnedReoriented=np.flipud(zBinned)

		data[zt] = pd.Series(zBinnedReoriented.ravel())

	df_tmp = pd.DataFrame(data=data)

	#most common zone by area
	grid[zname]=df_tmp[zonetypes[zname]].idxmax(axis=1)

	print(zname)

grid.to_pickle(params.geopandasDataDir + "AEZ.pkl")

title="Thermal Zone"
label="Thermal zone class 0 through 10 in each ~2 degree square cell"
Plotter.plotMap(grid,'thz',title,label,'thzZone',True)

title="Moisture Zone"
label="Moisture zone class 0 through 7 in each ~2 degree square cell"
Plotter.plotMap(grid,'mst',title,label,'mstZone',True)
