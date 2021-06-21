'''''
This code imports a raster of cropland area, roughly between mid-2000s and 2014. The original data comes from :
https://data.apps.fao.org/map/catalog/srv/eng/catalog.search#/metadata/ba4526fd-cdbf-4028-a1bd-5a559c4bff38

Downloaded the data from here:
http://www.fao.org/land-water/land/land-governance/land-resources-planning-toolbox/category/details/en/c/1036355/

Documentation for cropland data:
http://www.fao.org/uploads/media/glc-share-doc.pdf

Morgan Rivers
morgan@allfed.info
6/6/21
'''

from src import utilities
from src import params  # get file location and varname parameters for data import
from src.plotter import Plotter

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


#load the params from the params.ods file into the params object
params.importIfNotAlready()


ldata=rasterio.open(params.growAreaDataLoc)

print('reading grow area')
lArr=ldata.read(1)
print('done reading')

# we ignore the last latitude cell
lats = np.linspace(-90, 90 - params.latdiff, \
				   np.floor(180 / params.latdiff).astype('int'))
lons = np.linspace(-180, 180 - params.londiff, \
				   np.floor(360 / params.londiff).astype('int'))

latbins=np.floor(len(lArr)/len(lats)).astype('int')
lonbins=np.floor(len(lArr[0])/len(lons)).astype('int')

lArrResized=lArr[0:latbins*len(lats),0:lonbins*len(lons)]
sizeArray=[len(lats),len(lons)]
lBinned= utilities.rebin(lArrResized, sizeArray)
lBinnedReoriented=np.fliplr(np.transpose(lBinned))

lats2d, lons2d = np.meshgrid(lats, lons)

data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel()),
		# average fraction crop area.
		"fraction": pd.Series(lBinnedReoriented.ravel())/100.0}

df = pd.DataFrame(data=data)
geometry = gpd.points_from_xy(df.lons, df.lats)
gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)

grid= utilities.makeGrid(gdf)

grid.to_pickle(params.geopandasDataDir + "CropGrowFraction.pkl")

#1e4 meters to a hectare
grid['cellArea']=grid.to_crs({'proj':'cea'})['geometry'].area/1e4 

print('Earth Surface Area, billions of hectares')
print(grid['cellArea'].sum()/1e9)

#grid area in meters -> 1e6 meters per km^2, 1e2 km^2 per hectare
grid['growArea']=grid['cellArea']*grid['fraction']
grid.to_pickle(params.geopandasDataDir + "CropGrowHectares.pkl")

print('Total Crop Area, billions of hectares')
print(grid['growArea'].sum()/1e9)

plotGrowArea=True

label="Fraction Land for Crop Growing"
title="Crop Growing Area Fraction Between Years 2000-2014"
Plotter.plotMap(grid,'fraction',title,label,'CropGrowFraction',plotGrowArea)

label="Land Area for Crop Growing Each Cell (Ha)"
title="Crop Growing Area Between Years 2000-2014"
Plotter.plotMap(grid,'growArea',title,label,'CropGrowFraction',plotGrowArea)