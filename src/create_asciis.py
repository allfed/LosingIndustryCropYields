'''

Create a bunch of nice ascii files to look at before you run the stats

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
from src import utilities


from sys import platform
if platform == "linux" or platform == "linux2":
	#this is to ensure Morgan's computer doesn't crash
	import resource
	rsrc = resource.RLIMIT_AS
	resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb


params.importAll()

# fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv')
# utilities.create5minASCII(fertilizer,'n',params.asciiDir+'fertilizer')

# tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighReswhea.csv')
# utilities.create5minASCII(tillage,'whea_is_mech',params.asciiDir+'tillageWheat')

# pesticides=pd.read_csv(params.geopandasDataDir + 'WheatPesticidesHighRes.csv')
# utilities.create5minASCII(pesticides,'total_H',params.asciiDir+'pesticidesWheat')

# manure=pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv')
# utilities.create5minASCII(manure,'applied',params.asciiDir+'manure')

aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')
utilities.create5minASCII(aez,'soil',params.asciiDir+'aezSoil')
