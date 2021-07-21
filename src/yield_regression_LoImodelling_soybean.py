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

#import yield geopandas data for soybean

soyb_yield=pd.read_pickle(params.geopandasDataDir + 'SOYBCropYield.pkl')


#display first 5 rows of soybean yield dataset
soyb_yield.head()

#select all rows from soyb_yield for which the column growArea has a value greater than zero
soyb_nozero=soyb_yield.loc[soyb_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
soyb_kgha=soyb_nozero['yieldPerArea']
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
smean=soyb_kgha.mean()
smeadian=soyb_kgha.median()
ssd=soyb_kgha.std()
svar=soyb_kgha.var()
#logarithmize the values
soyb_kgha_log=np.log(soyb_kgha)

#plot soybean yield distribution in a histogram
plt.hist(soyb_kgha, bins=[1,50, 100, 175, 250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500])
plt.title('Soybean yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yieldPerArea
plt.hist(soyb_kgha_log, bins=[0,1,2,3,4,5,6,7,8,9,10,11])

#test if area without zeros aligns with FAOSTAT harvested area
soyb_area_ha = sum(soyb_nozero['growArea'])
print(soyb_area_ha)
#65436615.57129341
#102767896 FAOSTAT data for 2010

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