'''

File containing the code to explore data and perform a multiple regression
on yield for maize
'''

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src import params
from src import outdoor_growth
from src.outdoor_growth import OutdoorGrowth
from src import stat_ut
import pandas as pd
import scipy
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import resource
from sys import platform
if platform == "linux" or platform == "linux2":
    #this is to ensure Morgan's computer doesn't crash
    import resource
    rsrc = resource.RLIMIT_AS
    resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb

params.importAll()

'''
Import data, extract zeros and explore data statistic values and plots 
'''
maize_yield=pd.read_csv(params.geopandasDataDir + 'MAIZCropYieldHighRes.csv')

dmaize_raw = pd.DataFrame(data=maize_yield)
del maize_yield
print('filter > 100 ha')
dm0_elim=dmaize_raw[~(dmaize_raw['growArea'] > 100)]
# dm0_elim=dmaize_raw.drop(dmaize_raw['growArea'] > 100,axis==0)

raw_data=pd.read_csv(params.geopandasDataDir + 'FracIrrigationAreaHighRes.csv')
raw_data.drop(dm0_elim.index, inplace=True)

print("number of irrigated cells over 100 ha")
print(len(raw_data.loc[raw_data['fraction']>0]))
print("mean fraction area irrigated of these cells")
print(np.mean(raw_data.loc[raw_data['fraction']>0]['fraction']))
print("number of rainfed cells over 100 ha")
print(len(raw_data.loc[raw_data['fraction']==0]))
print("number of cells over 100 ha with no data for irrigation")
print(len(raw_data.loc[raw_data['fraction']<0]))

irr = raw_data['fraction']>0
rf = raw_data['fraction']==0

all_files = [\
    'MAIZCropYield',\
    'CornPesticides',\
    'Fertilizer',\
    'FertilizerManure',\
    'FracIrrigationArea',\
    'FracCropArea',\
    'TillageAllCrops',\
    'AEZ',\
    'FracReliant']


for f in all_files:
    print('reading '+ f)
    raw_data=pd.read_csv(params.geopandasDataDir + f+'HighRes.csv')
    raw_data.drop(dm0_elim.index, inplace=True)
    
    raw_data_irr = raw_data.loc[irr]
    raw_data_irr.to_csv(params.geopandasDataDir + f+"IrrFiltered.csv")

    raw_data_rf = raw_data.loc[rf==0]

    raw_data_rf.to_csv(params.geopandasDataDir + f+"RfFiltered.csv")




    print('filtered ' + f)
    print('')
    # del filtered_data
    del raw_data
    # quit()


    # #We do a weighted mean of a given resolution cell where values exist.
    # # if the added together areas fall below 100 hectares in the larger cell, drop it

    # #set no data values (-9) to zero area
    # np.where()

    # #multiply area by the value, and divide by total area within the group of cells. This gives a mean weighted by area
    

    # quit()
    # # create a grid
