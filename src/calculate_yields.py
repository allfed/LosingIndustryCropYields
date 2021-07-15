'''

An example file to deal with variables from different pkl files.
'''

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from src import params
from src.plotter import Plotter
from src import outdoor_growth
from src.outdoor_growth import OutdoorGrowth
import pandas as pd
import geopandas as gpd


params.importAll()

#total solar flux at surface , W/m^2
maize_yield=pd.read_pickle(params.geopandasDataDir + 'MAIZCropYield.pkl')
#total solar flux at surface , W/m^2
fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl')
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')
livestock=pd.read_pickle(params.geopandasDataDir + 'Livestock.pkl')
print(livestock.columns)
print(livestock.head())
# print(irrigation.columns)
# print(fertilizer.columns)
outdoorGrowth=OutdoorGrowth()
outdoorGrowth.correctForFertilizerAndIrrigation(maize_yield,fertilizer,irrigation)