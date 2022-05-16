'''''
Import fermanv1 pesticide database. Adds up each pesticide type for each crop.
Low and high bounds are listed.

the unit is kilogram per hectare per year (kg/ha-year) application rate

https://sedac.ciesin.columbia.edu/data/set/ferman-v1-pest-chemgrids-v1-01/metadata:
    Bounding Coordinates:

        West Bounding Coordinate: -180.000000
        East Bounding Coordinate: 180.000000
        North Bounding Coordinate: 56.000000
        South Bounding Coordinate: -84.000000

Horizontal Coordinate System Definition:

    Geographic:

        Latitude Resolution: 5.000000
        Longitude Resolution: 5.000000
        Geographic Coordinate Units: Arc-Minutes

    Geodetic Model:

        Horizontal Datum Name: WGS84
        Ellipsoid Name: WGS84
        Semi-major Axis: 6378137.000000
        Denominator of Flattening Ratio: 298.257224


Morgan Rivers
morgan@allfed.info
7/24/21
'''
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src import params  # get file location and varname parameters for data import
from src.plotter import Plotter

import numpy as np
import pandas as pd
import geopandas as gpd
import utilities
import rasterio
#load the params from the params.ods file into the params object
params.importIfNotAlready()
#import resource


MAKE_GRID = False

years=['2015']
# bounds=['L','H']
bounds=['H']
crops=[ \
    # 'Alfalfa',\
    'Corn',\
    # 'Cotton',\
    # 'OrcGra',\
    # 'Other',\
    # 'PasHay',\
    'Rice',\
    'Soybean',\
    # 'VegFru',\
    'Wheat'\
]
pesticides=[ \
    '2,4-d', #this is a more common pesticide\
    '2,4-db', #this is a more common pesticide\
    'Acephate',\
    'Acetochlor',\
    'Acifluorfen',\
    'Alachlor', #this is a more common pesticide\
    'Aminopyralid',\
    'Atrazine',\
    'Azoxystrobin',\
    'Bacillusamyloliquifacien',\
    'Bensulide',\
    'Bentazone',\
    'Bifenthrin',\
    'Bromoxynil',\
    'Calciumpolysulfide',\
    'Captan',\
    'Carbaryl', #this is a more common pesticide\
    'Chloropicrin',\
    'Chlorothalonil',\
    'Chlorpyrifos', #this is a more common pesticide\
    'Clethodim',\
    'Clomazone',\
    'Clopyralid',\
    'Clothianidin',\
    'Copperhydroxide',\
    'Coppersulfate',\
    'Coppersulfatetribasic',\
    'Cyhalofop',\
    'Cyhalothrin-lambda',\
    'Dicamba',\
    'Dichloropropene',\
    'Dichlorprop',\
    'Dicrotophos',\
    'Dimethenamid(-p)',\
    'Dimethoate',\
    'Diuron',\
    'Eptc', #this is a more common pesticide
    'Ethalfluralin',\
    'Ethoprophos',\
    'Flumioxazin',\
    'Fluometuron',\
    'Fluroxypyr',\
    'Flutolanil',\
    'Fomesafen',\
    'Glufosinate',\
    'Glyphosate',\
    'Halosulfuron',\
    'Hexazinone',\
    'Imazapyr',\
    'Imazethapyr',\
    'Imidacloprid',\
    'Indoxacarb',\
    'Isoxaflutole',\
    'Malathion', #this is a more common pesticide\
    'Mancozeb',\
    'Mcpa',\
    'Mesotrione',\
    'Metam',\
    'Metampotassium',\
    'Metconazole',\
    'Methylbromide',\
    'Metolachlor(-s)', #this is a more common pesticide\
    'Metribuzin',\
    'Metsulfuron',\
    'Msma',\
    'Oxyfluorfen',\
    'Paraquat',\
    'Pendimethalin',\
    'Petroleumoil',\
    'Phorate',\
    'Phosmet',\
    'Picloram',\
    'Pinoxaden',\
    'Prometryn',\
    'Propanil',\
    'Propargite',\
    'Propazine',\
    'Propiconazole',\
    'Prothioconazole',\
    'Pyraclostrobin',\
    'Pyroxasulfone',\
    'Quinclorac',\
    'Saflufenacil',\
    'Sethoxydim',\
    'Simazine',\
    'Sulfentrazone',\
    'Tebuconazole',\
    'Terbufos',\
    'Thiobencarb',\
    'Thiophanate-methyl',\
    'Tri-allate'#,\
    'Triclopyr',\
    'Trifloxystrobin',\
    'Trifluralin',\
    'Ziram'\
]

mn_lat=-56.00083
mx_lat=83.99917
mn_lon=-178.875
mx_lon=179.875


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

result=np.full((nbins*len(lats),nbins*len(lons)),-9.)

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
y="2015"
for c in crops:
    for b in bounds:
        print('')
        print(c+'_'+b+':')
        cSums=[]
        for p in pesticides:
            fn=params.pesticidesDataLoc+'APR_'+\
                c+\
                '_'+p+\
                '_'+y+\
                '_'+b+\
                '.tif'
            #if this combination is available
            if(os.path.exists(fn)):
                pdata=rasterio.open(fn)
                pArr=pdata.read(1)
                # strout=''
                # for a in pArr[1000:1100][90]:
                #   strout = strout + ' ' + str(a)
                # print(strout)
                # quit()

                print(pArr)
                result[start_lat_index:start_lat_index+len(pArr),start_lon_index:start_lon_index+len(pArr[0])]=pArr

                if(MAKE_GRID):
                    pArrResized=result[0:nbins*len(lats),0:nbins*len(lons)]

                    # pArrResizedAllNan=np.where(pArrResized<0, 1, pArrResized)
                    # pArrResizedFiltered=np.where(pArrResized<0, 0, pArrResized)
                    
                    #print pesticide if data for application of this pesticide exists 
                    print('    '+p)
                    
                    #record the pesticide amount for each pesticide
                    pBinned= utilities.rebin(pArrResized, sizeArray)
                    pBinnedReoriented=np.flipud(pBinned)

                    grid[p+'_'+b]=pd.Series(pBinnedReoriented.ravel())
                else: 
                    # df[p+'_'+b]=pd.Series(result.ravel())
                    pass

                #add the pesticides of this type for this crop to the total
                if(len(cSums)==0):
                    cSums = np.where(result<0,0,result)
                    mask = np.where(result<0,0,1)
                    # strout = ''
                    # for a in result[1000:1100][90]:
                    #   strout = strout + ' ' + str(a)
                    # print(strout)
                    # quit()

                    # for if we want to go back to rebinning at some point
                    # cSums=pArrResized
                    # mask = np.where(pArrResized<0,0,1)
                else:
                    #only 1 or nan. If ever a pesticide at a grid cell is not 
                    #nan, this value is not nan either.
                    #value of 1 indicates that there was a nonnegative value for some pesticide for this crop
                    mask = np.bitwise_or(mask,np.where(result<0,0,1))

                    cSums=np.array(cSums)+np.where(result<0,0,result)
                    # cSums=np.where(result<0,0,result)+np.array(result)

        if(len(cSums)==0):
            continue

        mask_final=np.where(mask==0,np.nan,1)
        
        #we get an appropriately sized array that also has nan values where there were only negatives in the original data.
        # cBinned= utilities.rebin(np.multiply(cSums,mask_final), sizeArray)
        cBinned= np.multiply(cSums,mask_final)
        
        if(MAKE_GRID):
            cBinnedReoriented=np.flipud(cBinned)
            grid['total_'+b]=pd.Series(cBinnedReoriented.ravel())
        else:
            cBinnedReoriented=np.fliplr(np.transpose(cBinned))
            df['total_'+b]=pd.Series(cBinnedReoriented.ravel())


    if(MAKE_GRID):
        grid.to_csv(params.geopandasDataDir + c + "Pesticides.csv")
        plotGrowArea=True
        title=c+" Total Pesticide Application Rate, 2015, Lower Bound"
        label="Application Rate (kg/ha/year)"
        Plotter.plotMap(grid,'total_L',title,label,'TotPesticidesByCropLow',plotGrowArea)
        plotGrowArea=True
        title=c+" Total Pesticide Application Rate, 2015, Upper Bound"
        label="Application Rate (kg/ha/year)"
        Plotter.plotMap(grid,'total_H',title,label,'TotPesticidesByCropHigh',plotGrowArea)
        # title="2,4-d Pesticide Application Rate, 2020, Upper Bound"
        # label="Application Rate (kg/ha/year)"
        # Plotter.plotMap(grid,'2,4-d_total_H',title,label,'CropYield',plotGrowArea)
    else:
        print(df.columns)
        df.to_csv(params.geopandasDataDir + c + "PesticidesHighRes.csv")
        