'''
This code imports and downsamples a raster (geotiff) of yield gaps
from GAEZ v4 data portal. 

Map was downloaded by clicking on  crop specific achievable yield ratio for maize, wheat, soybean and wetland rice.
mze_2010_yga_cl.tif,whe_2010_yga_cl ... etc.

For description, found the link in this document
http://www.fao.org/3/cb5167en/cb5167en.pdf
which links here
https://gaez.fao.org/pages/glossary

and gives you the symbology yield production gap clr file: 
https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/documentation/GAEZ4_symbology_files.zip


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
from src.utilities.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import src.utilities.utilities as utilities

#import resource
# import datetime
from sys import platform
if platform == "linux" or platform == "linux2":
    #this is to ensure Morgan's computer doesn't crash
    import resource
    rsrc = resource.RLIMIT_AS
    resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb


#load the params from the params.ods file into the params object
params.importIfNotAlready()

MAKE_GRID = False

crops = ['mze','rcw','soy','whe']

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
sizeArray=[len(lats),len(lons)]

# time5 = datetime.datetime.now()

print('reading yield gap data')




#based off the clr file:
# 0: NA
# 1: < 10%
# 2: 10%-25%
# 3: 25%-40%
# 4: 40%-55%
# 5: 55%-70%
# 6: 70%-85%
# 7: > 85%
yield_gap_value_dictionary={\
    0: -9,
    1: 0.05,
    2: 0.175,
    3: 0.325,
    4: 0.475,
    5: 0.625,
    6: 0.775,
    7: 0.925}

#create yield gap estimate for each crop
for c in crops:
    gdata=rasterio.open(params.aezDataLoc+c+'_2010_yga_cl.tif')
    gArr=gdata.read(1)

    gArrReoriented=np.fliplr(np.transpose(gArr))

    df['one_minus_gap'] = pd.Series(gArrReoriented.ravel())
    df['one_minus_gap'] = df['one_minus_gap'].map(yield_gap_value_dictionary)

    assert(df['lats'].iloc[-1]>df['lats'].iloc[0])
    assert(df['lons'].iloc[-1]>df['lons'].iloc[0])

    df.to_csv(params.geopandasDataDir + c+ "YieldGapHighRes.csv")