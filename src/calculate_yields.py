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
# import resource
# rsrc = resource.RLIMIT_AS
# resource.setrlimit(rsrc, (2e9, 2e9))#no more than 2 gb


params.importAll()

MAKE_GRID = False

if(MAKE_GRID):
	#total solar flux at surface , W/m^2
	maize_yield=pd.read_csv(params.geopandasDataDir + 'MAIZCropYield.csv')
	#total solar flux at surface , W/m^2
	fertilizer=pd.read_csv(params.geopandasDataDir + 'Fertilizer.csv')
else:
	maize_yield=pd.read_csv(params.geopandasDataDir + 'MAIZCropYieldHighRes.csv')

	fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv')

# irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation'+to_append+'.pkl')
# livestock=pd.read_pickle(params.geopandasDataDir + 'Livestock'+to_append+'.pkl')
# print(livestock.columns)
# print(livestock.head())
# print(irrigation.columns)
# print(fertilizer.columns)
outdoorGrowth=OutdoorGrowth()
outdoorGrowth.correctForFertilizer(maize_yield,fertilizer)