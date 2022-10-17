'''''
This code imports a raster of cell area.

The original data comes from the livestock dataset, glw3:
https://dataverse.harvard.edu/dataverse/glw_3?q=&types=files&sort=dateSort&order=desc&page=4


Morgan Rivers
morgan@allfed.info
6/6/21
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


#load the params from the params.ods file into the params object
params.importIfNotAlready()


five_minute = 5/60
# we ignore the last latitude cell
lats = np.linspace(-90, 90 - five_minute, \
                   np.floor(180 / five_minute).astype('int'))
lons = np.linspace(-180, 180 - five_minute, \
                   np.floor(360 / five_minute).astype('int'))


print('reading cell area')
# it's a bit wierd to pull from another dataset, but the livestock dataset
# conveniently had the area of each 5 minute raster cell available
# below we use the area of each cell to estimate the total area tilled in km^2
cdata=rasterio.open(params.livestockDataLoc+'8_Areakm.tif')
cArr=np.transpose(cdata.read(1))
print('done reading')



lats2d, lons2d = np.meshgrid(lats, lons)

#the area data is in km^2 percent, so we multiply by 100 to get hectares
data = {"lats": pd.Series(lats2d.ravel()),
        "lons": pd.Series(lons2d.ravel()),
        # average fraction crop area.
        "area": pd.Series(cArr.ravel())*100.0}

df = pd.DataFrame(data=data)
# print(len(df['']))

#fraction of cell that's cropland
df.to_csv(params.geopandasDataDir + "CellAreaHighRes.csv")
