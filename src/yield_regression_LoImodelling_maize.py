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

#import yield geopandas data for maize

maize_yield=pd.read_pickle(params.geopandasDataDir + 'MAIZCropYield.pkl')

#display first 5 rows of maize yield dataset
maize_yield.head()

#select all rows from maize_yield for which the column growArea has a value greater than zero
maize_nozero=maize_yield.loc[maize_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
maize_kgha=maize_nozero['yieldPerArea']
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
mmean=maize_kgha.mean()
mmeadian=maize_kgha.median()
msd=maize_kgha.std()
mvar=maize_kgha.var()
#check the datatype of yieldPerArea and logarithmize the values
maize_kgha.dtype
maize_kgha_log=np.log(maize_kgha)

#plot maize yield distribution in a histogram
plt.hist(maize_kgha, bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yieldPerArea
plt.hist(maize_kgha_log, bins=[0,1,2,3,4,5,6,7,8,9,10,11])

#maize yield histogram still contains many low values: do two more extractions of columns, one excluding all zeros in yield
#and one excluding all cells which contain zeros for totalYield (does totalYield equal total production?)

#yield excluding zeros in total yield
maize_zeroarea=maize_nozero.loc[maize_nozero['totalYield'] > 0]
maize_kgha_area=maize_zeroarea['yieldPerArea']

#yield excluding zeros in yieldPerArea
maize_zeroyield=maize_nozero.loc[maize_nozero['yieldPerArea'] > 0]
maize_kgha_yield=maize_zeroyield['yieldPerArea']

#both operations don't change the number of selected rows

#test if area without zeros aligns with FAOSTAT harvested area
maize_area_ha = sum(maize_nozero['growArea'])
print(maize_area_ha)
#164569574.0937798
#164586904	#FAOSTAT area from 2010 for maize

#subplot for all histograms
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(maize_yield['yieldPerArea'], bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 0].hist(maize_kgha, bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[0, 1].hist(maize_kgha_area, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 1].hist(maize_kgha_yield, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])


plt.hist(maize_kgha_area, bins=[1,50, 100, 175, 250, 375, 500,625,750,875,1000,1125,1250,1375,1500,1625,1750,1875,2000,2500,3000,3500,4000,4500,5000,5500,6000,7000,8000,9000,10000,11000,12000])
print(maize_kgha_yield)



'''
fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl')
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')

print(fertilizer.columns)
print(fertilizer.head())
# print(irrigation.columns)
# print(fertilizer.columns)
outdoorGrowth=OutdoorGrowth()
outdoorGrowth.correctForFertilizerAndIrrigation(maize_yield,fertilizer,irrigation)


saved from the outdoorGrowth file which we used to discuss an example case for appending datasets based on a condition
	#now iterate through rows and calculate values for each grid cell
		corrected=[0]*len(fertilizer)
     #   dependend=[]
     #  independent=[]
		for index,row in fertilizer.iterrows():
			
			irr_val_tot = irrigation.iloc[index]['area']
			fer_nit_val = row['n']
			yield_val = yields.iloc[index]['totalYield']
            
     #       if(area_wheat_val>0):
      #          dependent.append(yield_val)
       #         independent.append(fer_nit_val)
'''    