'''''
Import manure fertilizer. Adds up each 

the unit is kilogram nitrogen per km^2 per year (kg/km^2/year) application rate

https://doi.pangaea.de/10.1594/PANGAEA.871980
https://essd.copernicus.org/articles/9/667/2017/essd-9-667-2017.pdf

metadata:
    ncols         4320
    nrows         2124
    xllcorner     -180
    yllcorner     -88.5
    cellsize      0.0833333
    NODATA_value  -9999

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
from src import utilities
import rasterio




#load the params from the params.ods file into the params object
params.importIfNotAlready()


mn_lat=-88.5
mx_lat=88.5
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
files=['appliedNyy2014.asc','producedNyy2014.asc']
for f in files:
    coltitle=f.split('N')[0] #either 'applied' or 'produced'
    fdata=rasterio.open(params.manureFertilizerDataLoc+f)
    fArr=fdata.read(1)

    result[start_lat_index:start_lat_index+len(fArr),start_lon_index:start_lon_index+len(fArr[0])]=fArr


    fArrResized=result[0:nbins*len(lats),0:nbins*len(lons)]

    # fArrResizedFiltered=np.where(fArrResized<0, 0, fArrResized)
    
    #record the nitrogen amount for each pesticide
    fBinned= utilities.rebin(fArrResized, sizeArray)
    
    if(MAKE_GRID):
        fBinnedReoriented=np.flipud(fBinned)
        grid[coltitle]=pd.Series(fBinnedReoriented.ravel())
    else:
        fBinnedReoriented=np.fliplr(np.transpose(fBinned))
        df[coltitle]=pd.Series(fBinnedReoriented.ravel())


if(MAKE_GRID):
    grid.to_csv(params.geopandasDataDir + "FertilizerManure.csv")
        
    plotGrowArea=True
    title=" Total Manure Nitrogen Application Rate, 2014"
    label="Application Rate (kg/km^2/year)"
    Plotter.plotMap(grid,'applied',title,label,'TotManureNApplied',plotGrowArea)
    plotGrowArea=True
    title=" Total Manure Nitrogen Production Rate, 2014"
    label="Application Rate (kg/km^2/year)"
    Plotter.plotMap(grid,'produced',title,label,'TotManureNProduced',plotGrowArea)
    # title="2,4-d Pesticide Application Rate, 2020, Upper Bound"
    # label="Application Rate (kg/ha/year)"
    # Plotter.plotMap(grid,'2,4-d_total_H',title,label,'CropYield',plotGrowArea)
else:
    df.to_csv(params.geopandasDataDir + "FertilizerManureHighRes.csv")