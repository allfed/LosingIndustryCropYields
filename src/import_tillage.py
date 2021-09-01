'''''
This code imports and downsamples a netcdf defined set of arrays of tillage

https://essd.copernicus.org/articles/11/823/2019/

The paper states that low income, not irrigated, or small farm size are 
generally less likely to be mechanized. It states the assumption that the 
minimum depth of mechanized tillage is 20cm. It assumes that conventional 
annual tillage is always mechanized. 

1 = conventional annual tillage
2 = traditional annual tillage
3 = reduced tillage
4 = Conservation Agriculture
5 = rotational tillage
6 = traditional rotational tillage

1. conventional annual tillage (MECHANIZED)

2. traditional annual tillage:
	annual, small field size, poor area (NOT MECHANIZED)

3. reduced tillage : always <20cm (NOT MECHANIZED)

4. conservation agriculture:
	no tillage assumed (NOT MECHANIZED)

5. rotational tillage:
	not annual crop, field >2ha, soil >=20cm deep (MECHANIZED)

6. traditional rotational tillage:
	soil >=15cm, not annual crop, field size <2hr, GNI small (NOT MECHANIZED)

Finally, we use the areakm data for each cell to determine total size of each
5 minute grid cell.

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
from src import utilities
import netCDF4 as nc
import rasterio

import resource

rsrc = resource.RLIMIT_AS
resource.setrlimit(rsrc, (3e9, 3e9))#no more than 2 gb

#load the params from the params.ods file into the params object
params.importIfNotAlready()
ds=nc.Dataset(params.tillageDataLoc)

mn_lat=-56
mx_lat=84
mn_lon=-180
mx_lon=180

MAKE_GRID = False

pSums={}
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
# result=np.full((nbins*len(lats),nbins*len(lons)),np.nan)

lats2d, lons2d = np.meshgrid(lats, lons)
data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel())}
df = pd.DataFrame(data=data)

if(MAKE_GRID):

	#make geometry
	geometry = gpd.points_from_xy(df.lons, df.lats)
	gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
	grid= utilities.makeGrid(gdf)

sizeArray=[len(lats),len(lons)]

crops=['whea','rice','maiz','soyb']

#it's a bit wierd to pull from another dataset, but the livestock dataset
# conveniently had the area of each 5 minute raster cell available
#below we use the area of each cell to estimate the total area tilled in km^2
adata=rasterio.open(params.livestockDataLoc+'8_Areakm.tif')
aArr=adata.read(1)


for c in crops:
	arr=np.array(ds[c+'_till'])
	for m in ['mech','non_mech']:

		#element-wise or,if any are true, then cell value is true
		if(m=='mech'): #mechanized: 1 or 5
			# mask=np.where(np.isnan(arr),np.nan,0)
			mech_areas=np.bitwise_or(arr==1, arr==5)
			
			# areas=np.where(mech_areas,1,np.nan)+mask
			areas=np.where(mech_areas,1,0)
		else: #non mechanized: 2, 3, 4, or 6
			non_mech_areas=np.bitwise_or(np.bitwise_or(np.bitwise_or(arr==2,arr==3),arr==4),arr==6)
			areas=np.where(non_mech_areas,1,0)
		result[start_lat_index:start_lat_index+len(areas),start_lon_index:start_lon_index+len(areas[0])]=areas

		cArrResized=result[0:nbins*len(lats),0:nbins*len(lons)]

		# cArrResized.fill(True)
		grid_area=np.multiply(cArrResized,aArr)
		# print(grid_area)
		# quit()
		cBinned= utilities.rebinCumulative(grid_area, sizeArray)
		cBinnedReoriented=np.flipud(cBinned)

		if(MAKE_GRID):
			grid[c+'_'+m]=pd.Series(cBinnedReoriented.ravel())
		else:
			df[c+'_'+m]=pd.Series(cBinnedReoriented.ravel())

		# print(grid[c+'_'+m])
		# quit()
	if(MAKE_GRID):
		grid[c+'_is_mech_tmp']=grid[c+'_mech']>=grid[c+'_non_mech']
		grid[c+'_is_not_mech_tmp']=grid[c+'_mech']<grid[c+'_non_mech']
		grid[c+'_no_crops']=(grid[c+'_mech']==0) & (grid[c+'_non_mech']==0)
		grid[c+'_is_mech'] = np.where(grid[c+'_no_crops'],np.nan,grid[c+'_is_mech_tmp'])
		del grid[c+'_is_not_mech_tmp']
		del grid[c+'_is_mech_tmp']

		plotGrowArea=True
		title=c+" Mechanized Tillage area, 2005"
		label="Tillage is mechanized"
		Plotter.plotMap(grid,c+'_is_mech',title,label,'TillageMechWheat',plotGrowArea)
	else:
		df[c+'_is_mech_tmp']=df[c+'_mech']>=df[c+'_non_mech']
		df[c+'_is_not_mech_tmp']=df[c+'_mech']<df[c+'_non_mech']
		df[c+'_no_crops']=(df[c+'_mech']==0) & (df[c+'_non_mech']==0)
		df[c+'_is_mech'] = np.where(df[c+'_no_crops'],np.nan,df[c+'_is_mech_tmp'])
		del df[c+'_is_not_mech_tmp']
		del df[c+'_is_mech_tmp']

# Plotter.plotMap(grid,'whea_is_not_mech',title,label,'TillageMechWheat',plotGrowArea)
if(MAKE_GRID):
	grid.to_csv(params.geopandasDataDir + "Tillage.csv")
else:
	df.to_csv(params.geopandasDataDir + "TillageHighRes.csv")