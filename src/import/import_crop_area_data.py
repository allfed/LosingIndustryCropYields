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
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utilities import params  # get file location and varname parameters for data import
from utilities.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import utilities.utilities as utilities

#load the params from the params.ods file into the params object
params.importIfNotAlready()


ldata=rasterio.open(params.cropAreaDataLoc)

#it's a bit wierd to pull from another dataset, but the livestock dataset
# conveniently had the area of each 5 minute raster cell available
#below we use the area of each cell to estimate the total area tilled in km^2
adata=rasterio.open(params.livestockDataLoc+'8_Areakm.tif')
aArr=adata.read(1)

print('reading grow area')
lArr=ldata.read(1)
print('done reading')
five_minute = 5/60
# we ignore the last latitude cell
lats = np.linspace(-90, 90 - five_minute, \
                   np.floor(180 / five_minute).astype('int'))
lons = np.linspace(-180, 180 - five_minute, \
                   np.floor(360 / five_minute).astype('int'))

latbins=np.floor(len(lArr)/len(lats)).astype('int')
lonbins=np.floor(len(lArr[0])/len(lons)).astype('int')

lArrResized=lArr[0:latbins*len(lats),0:lonbins*len(lons)]
sizeArray=[len(lats),len(lons)]
lBinned= utilities.rebin(lArrResized, sizeArray)
lBinnedReoriented=np.fliplr(np.transpose(lBinned))


#it's a bit wierd to pull from another dataset, but the livestock dataset
# conveniently had the area of each 5 minute raster cell available
#below we use the area of each cell to estimate the total area tilled in km^2
# adata=rasterio.open(params.livestockDataLoc+'8_Areakm.tif')
# aArr=np.transpose(adata.read(1))


# grid_area=np.multiply(lBinnedReoriented,aArr)


lats2d, lons2d = np.meshgrid(lats, lons)

#the crop data is a percent, so we divide by 100 to get a fraction
data = {"lats": pd.Series(lats2d.ravel()),
        "lons": pd.Series(lons2d.ravel()),
        # average fraction crop area.
        "fraction": pd.Series(lBinnedReoriented.ravel())/100.0}

df = pd.DataFrame(data=data)
# print(len(df['']))

#fraction of cell that's cropland
df.to_pickle(params.inputDataDir + "FracCropAreaHighRes.pkl", compression='zip')

# geometry = gpd.points_from_xy(df.lons, df.lats)
# gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)

# grid= utilities.makeGrid(gdf)
# utilities.saveAsCSV('CropGrowFraction',grid)
# grid.to_pickle(params.geopandasDataDir + "CropGrowFraction.pkl")

#1e4 meters to a hectare
# grid['cellArea']=grid.to_crs({'proj':'cea'})['geometry'].area/1e4 

# print('Earth Surface Area, billions of hectares')
# print(grid['cellArea'].sum()/1e9)

# grid['growArea']=grid['cellArea']*grid['fraction']
# utilities.saveAsCSV('CropGrowHectares',grid)
# grid.to_pickle(params.geopandasDataDir + "CropGrowHectares.pkl")

# print('Total Crop Area, billions of hectares')
# print(grid['growArea'].sum()/1e9)

# plotGrowArea=False

# label="Fraction Land for Crop Growing"
# title="Crop Growing Area Fraction Between Years 2000-2014"
# Plotter.plotMap(grid,'fraction',title,label,'CropGrowFraction',plotGrowArea)

# label="Land Area for Crop Growing Each Cell (Ha)"
# title="Crop Growing Area Between Years 2000-2014"
# Plotter.plotMap(grid,'growArea',title,label,'CropGrowFraction',plotGrowArea)