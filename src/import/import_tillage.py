'''''
This code imports and downsamples a netcdf defined set of arrays of tillage

https://essd.copernicus.org/articles/11/823/2019/

The paper states that low income, not irrigated, or small farm size are 
generally less likely to be mechanized. It states the assumption that the 
minimum depth of mechanized tillage is 20cm. It assumes that conventional 
annual tillage is always mechanized. 

nan = no data
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
    no tillage assumed (MECHANIZED)

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
import netCDF4 as nc

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

allcrops = pd.DataFrame()
for c in crops:
    arr=np.array(ds[c+'_till'])

    print('imported array')


    for m in ['mech','non_mech']:

        #element-wise or,if any are true, then cell value is true
        #any nan values will be false for both mech and non-mech
        if(m=='mech'): #mechanized: 1, 4 or 5
            mech_areas=np.bitwise_or(np.bitwise_or(arr==1, arr==4), arr==5)
            
            # areas=np.where(mech_areas,1,np.nan)+mask
            areas=np.where(mech_areas,1,0)
        else: #non mechanized: 2, 3, or 6
            non_mech_areas=np.bitwise_or(np.bitwise_or(arr==2,arr==3),arr==6)
            areas=np.where(non_mech_areas,1,0)
        result[start_lat_index:start_lat_index+len(areas),start_lon_index:start_lon_index+len(areas[0])]=areas

        cArrResized=result[0:nbins*len(lats),0:nbins*len(lons)]

        # cArrResized.fill(True)
        grid_area=np.multiply(cArrResized,aArr)
        # print(grid_area)
        # quit()


        cBinned= utilities.rebinCumulative(grid_area, sizeArray)

        if(MAKE_GRID):
            cBinnedReoriented=np.flipud(cBinned)
            grid[c+'_'+m]=pd.Series(cBinnedReoriented.ravel())
        else:
            cBinnedReoriented=np.fliplr(np.transpose(cBinned))
            df[c+'_'+m]=pd.Series(cBinnedReoriented.ravel())

        # print(grid[c+'_'+m])
        # quit()

    #the idea here is previously we scaled the fraction of area of a crop that was mechanized by the total area of the cell (roughly 10km^2 at equator).We again scaled the fraction of area of a crop that was not mechanized by the area of the cell (roughly 10km^2 at equator). Then, we compared the relative amount of mechanized vs not mechanized to decide whether a grid cell was mechanized or not. It works, although it's unnecessarily complicated. The scaling doesn't actually help anything here when we're not rebinning.
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
        del df[c+'_mech']
        del df[c+'_non_mech']

    print('finished first round of mech and nonmech')

        # #no data is -9 for the ascii files.
        # df[c+'_is_mech'] = np.where(df[c+'_no_crops'],-9,df[c+'_is_mech_tmp'])
        # tillage_mech = df[c+'_is_mech']
        # # tillage_not_mech = df[c+'_is_mech']
        # tillage_mech.to_csv(params.geopandasDataDir + "TillageHighRes.csv")

    #here, we look at the crops of interest and assess whether there is tillage for any of these four crops. We are currently ignoring tillage for all the other crops, although this may not be the best strategy.
    if(not MAKE_GRID):
        if(len(allcrops) == 0):
            # mask[m] = df.where(df[m]<0,0)
            allcrops['mask'] = df[c+'_no_crops']
            allcrops['is_mech'] = df[c+'_is_mech'].where(df[c+'_is_mech']==1,0).astype('bool')
        else:
            #areas that are masked off as having no crops are 1, otherwise if there are crops, 0
            #anywhere there isn't a mask for any crop, there isn't a mask for all crops
            allcrops['mask'] = allcrops['mask'] & df[c+'_no_crops']

            #where any crops are tilled, this is 1.
            allcrops['is_mech'] = allcrops['is_mech'] | df[c+'_is_mech'].where(df[c+'_is_mech']==1,0).astype('bool')
        df[c+'_is_mech'] = np.where(np.isnan(df[c+'_is_mech']),-9,df[c+'_is_mech'])
        print('about to save tillage for crop '+c)
        df.to_csv(params.geopandasDataDir + "TillageHighRes"+c+".csv")
        del df[c+'_is_mech']
        del df[c+'_no_crops']

        # quit()
        print('tillage for crop saved')
        # has to be "no crops" for every crop type to be masked off as nan

print('masking operations')
to_save = pd.DataFrame()
df['is_mech'] = np.where(allcrops['mask'],-9,allcrops['is_mech'])
del allcrops
del cBinnedReoriented
print('save for all crops ')
df.to_csv(params.geopandasDataDir + "TillageAllCropsHighRes.csv")

# Plotter.plotMap(grid,'whea_is_not_mech',title,label,'TillageMechWheat',plotGrowArea)
if(MAKE_GRID):
    grid.to_csv(params.geopandasDataDir + "Tillage.csv")
