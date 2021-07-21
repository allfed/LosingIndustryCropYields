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
import scipy
import matplotlib.pyplot as plt
import numpy as np


params.importAll()

#import yield geopandas data for wheat

wheat_yield=pd.read_pickle(params.geopandasDataDir + 'WHEACropYield.pkl')

#display first 5 rows of wheat yield dataset
wheat_yield.head()

#select all rows from wheat_yield for which the column growArea has a value greater than zero
wheat_nozero=wheat_yield.loc[wheat_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
wheat_kgha=wheat_nozero['yieldPerArea']
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
wmean=wheat_kgha.mean()
wmeadian=wheat_kgha.median()
wsd=wheat_kgha.std()
wvar=wheat_kgha.var()
#logarithmize the values
wheat_kgha_log=np.log(wheat_kgha)

#plot soybean yield distribution in a histogram
plt.hist(wheat_kgha, bins=[1,50, 100, 175, 250,500,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,5000,6000,7000,8000,8500])
plt.title('Wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yieldPerArea
plt.hist(wheat_kgha_log, bins=[0,1,2,3,4,5,6,7,8,9,10,11])

#test if area without zeros aligns with FAOSTAT harvested area
wheat_area_ha = sum(wheat_nozero['growArea'])
print(wheat_area_ha)
#220717790.9382064
#215602998 #FAOSTAT data for 2010

'''
#subplot for all histograms
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(maize_yield['yieldPerArea'], bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 0].hist(maize_kgha, bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[0, 1].hist(maize_kgha_area, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 1].hist(maize_kgha_yield, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])








fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl')
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')

print(fertilizer.columns)
print(fertilizer.head())
# print(irrigation.columns)
# print(fertilizer.columns)
outdoorGrowth=OutdoorGrowth()
outdoorGrowth.correctForFertilizerAndIrrigation(maize_yield,fertilizer,irrigation)
'''
