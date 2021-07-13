'''

A quick utility file to create useful csvs. Each row of the csv is a 
different ~2 by 2 degree cell from the nuclear winter model.

'''

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from src import params
from src.plotter import Plotter
import pandas as pd
import geopandas as gpd

params.importAll()

#total solar flux at surface , W/m^2
maize_yield=pd.read_pickle(params.geopandasDataDir + 'MAIZCropYield.pkl')

print(maize_yield.head())