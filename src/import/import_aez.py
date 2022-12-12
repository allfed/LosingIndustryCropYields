'''
This code imports and downsamples a raster (geotiff) of agroecological zones 
(AEZ). 


Morgan Rivers
morgan@allfed.info
7/24/21
'''
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utilities import params  # get file location and varname parameters for data import
from utilities.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import utilities.utilities as utilities


#load the params from the params.ods file into the params object
params.importIfNotAlready()

MAKE_GRID = False

# aquastat=pd.read_csv(params.aquastatIrrigationDataLoc,index_col=False)
# aq=aquastat.dropna(how='all').replace('',np.nan)
AEZs = [
'mst_class_CRUTS32_Hist_8110_100_avg.tif',
'thz_class_CRUTS32_Hist_8110_100_avg.tif',
#source below is 30 arcsecond; we downsampled to soil_5arcmin.tif before import
# 'soil_regime_CRUTS32_Hist_8110.tif'
'soil_5arcmin.tif'
]

mn_lat=-90
mx_lat=90
mn_lon=-180
mx_lon=180

nbins=params.growAreaBins

#5 arcminutes in degrees
five_minute=5/60
# time1 = datetime.datetime.now()

start_lat_index=np.floor((90-mx_lat)/five_minute).astype('int')
start_lon_index=np.floor((mn_lon+180)/five_minute).astype('int')

# we ignore the last latitude cell
lats = np.linspace(-90, 90 - params.latdiff, \
                    np.floor(180 / params.latdiff).astype('int'))
lons = np.linspace(-180, 180 - params.londiff, \
                    np.floor(360 / params.londiff).astype('int'))

result=np.zeros((nbins*len(lats),nbins*len(lons)))
# time2 = datetime.datetime.now()

lats2d, lons2d = np.meshgrid(lats, lons)
data = {"lats": pd.Series(lats2d.ravel()),
        "lons": pd.Series(lons2d.ravel())}
# time3 = datetime.datetime.now()
df = pd.DataFrame(data=data)
# time4 = datetime.datetime.now()
if(MAKE_GRID):
    #make geometry
    geometry = gpd.points_from_xy(df.lons, df.lats)
    gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)
    grid= utilities.makeGrid(gdf)

sizeArray=[len(lats),len(lons)]

# time5 = datetime.datetime.now()

#it's a bit wierd to pull from another dataset, but the livestock dataset
# conveniently had the area of each 5 minute raster cell available
#below we use the area of each cell to estimate the relative weight of the zone 
#when comparing the most commmon zone in a region
adata=rasterio.open(params.livestockDataLoc+'8_Areakm.tif')
# time6 = datetime.datetime.now()
aArr=adata.read(1)
# time7 = datetime.datetime.now()

zonetypes={}
zonetypes['thz']=[0,1,2,3,4,5,6,7,8,9,10]
zonetypes['mst']=[0,1,2,3,4,5,6,7]
zonetypes['soil']=[0,1,2,3,4,5,6,7,8]
print('reading agroecological zone data')
df_tmp = pd.DataFrame(data=data)

for z in AEZs:
    zname=z.split('_')[0]
    # time8 = datetime.datetime.now()
    zdata=rasterio.open(params.aezDataLoc+z)
    # time9 = datetime.datetime.now()
    zArr=zdata.read(1)

    if(MAKE_GRID):
        # print(zArr[1500:1550,1400:1420])
        for zt in zonetypes[zname]:
            areas = (zArr==zt)

            # time8 = datetime.datetime.now()
            latbins=np.floor(len(areas)/len(lats)).astype('int')
            lonbins=np.floor(len(areas[0])/len(lons)).astype('int')
            zArrResized=areas[0:latbins*len(lats),0:lonbins*len(lons)]
            # time9 = datetime.datetime.now()

            grid_area=np.multiply(zArrResized,aArr)
            # time10 = datetime.datetime.now()

            zBinned= utilities.rebinCumulative(grid_area, sizeArray)
            # time11 = datetime.datetime.now()
            zBinnedReoriented=np.flipud(zBinned)
            # time12 = datetime.datetime.now()

            data[zt] = pd.Series(zBinnedReoriented.ravel())
            # time13 = datetime.datetime.now()

        print('done reading '+z)
        df_tmp = pd.DataFrame(data=data)
        # time14 = datetime.datetime.now()

        #most common zone by area
        grid[zname]=df_tmp[zonetypes[zname]].idxmax(axis=1)
    else:
        # time10 = datetime.datetime.now()
        zArrReoriented=np.fliplr(np.transpose(zArr))

        df[zname] = pd.Series(zArrReoriented.ravel())
        # time11 = datetime.datetime.now()


    print(zname)

if(MAKE_GRID):
    grid.to_csv(params.geopandasDataDir + "AEZ.csv")
    # time12 = datetime.datetime.now()

    title="Thermal Zone"
    label="Thermal zone class 0 through 10 in each ~2 degree square cell"
    Plotter.plotMap(grid,'thz',title,label,'thzZone',True)

    title="Moisture Zone"
    label="Moisture zone class 0 through 7 in each ~2 degree square cell"
    Plotter.plotMap(grid,'mst',title,label,'mstZone',True)

    title="Soil Zone"
    label="Soil zone class 0 through 8 in each ~2 degree square cell"
    Plotter.plotMap(grid,'soil',title,label,'soilZone',True)
else:
    assert(df['lats'].iloc[-1]>df['lats'].iloc[0])
    assert(df['lons'].iloc[-1]>df['lons'].iloc[0])

    df.to_csv(params.geopandasDataDir + "AEZHighRes.csv")
# print('time2: '+str((time2-time1).total_seconds() * 1000))
# print('time3: '+str((time3-time2).total_seconds() * 1000))
# print('time4: '+str((time4-time3).total_seconds() * 1000))
# print('time5: '+str((time5-time4).total_seconds() * 1000))
# print('time6: '+str((time6-time5).total_seconds() * 1000))
# print('time7: '+str((time7-time6).total_seconds() * 1000))
# print('time8: '+str((time7-time6).total_seconds() * 1000))
# print('time9: '+str((time7-time6).total_seconds() * 1000))
# print('time10: '+str((time7-time6).total_seconds() * 1000))
# print('time11: '+str((time7-time6).total_seconds() * 1000))
# print('time12: '+str((time7-time6).total_seconds() * 1000))
# print('time13: '+str((time7-time6).total_seconds() * 1000))
# print('time14: '+str((time7-time6).total_seconds() * 1000))
# print('time15: '+str((time7-time6).total_seconds() * 1000))
