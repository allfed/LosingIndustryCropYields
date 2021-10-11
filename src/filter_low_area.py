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

#downsample the data
#We do a weighted mean of a given resolution cell where values exist.
def downsample(raw_data,area_data):
    
    #first set no data values (-9) to effectively zero area
    area_data=np.where(raw_data<0, 0, area_data)

    #reshape the 1d array to a 2d array


    #multiply area of cell by the value being downsampled

    # "rebin", which means make the large 2d array into a smaller 2d array. Each element is the sum of the values times crop area in that cell

    #calculated total area within the group of cells

    #divide the weighted value by area by the total area within the group of cells. This gives a mean weighted by area.

    #make the 2d array back into a 1d array

    return downsampled
'''
Import data, extract zeros and explore data statistic values and plots 
'''
maize_yield=pd.read_csv(params.geopandasDataDir + 'MAIZCropYieldHighRes.csv')

dmaize_raw = pd.DataFrame(data=maize_yield)
del maize_yield
print('filter > 100 ha')
dm0_elim=dmaize_raw[~(dmaize_raw['growArea'] > 100)]
# dm0_elim=dmaize_raw.drop(dmaize_raw['growArea'] > 100,axis==0)

all_files = [\
    'mzeYieldGap',\
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

    # downsampled=downsample(raw_data,area_data)

    #drop all the cells with areas below the expected amount 
    raw_data.drop(dm0_elim.index, inplace=True)
    

    #remove this strange unnamed column
    raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('^Unnamed')]

    print(raw_data.index)
    print(raw_data)
    print(raw_data.columns)
    
    raw_data.index.name = 'index'
    raw_data.to_csv(params.geopandasDataDir + f+"Filtered.csv")
    print('filtered ' + f)
    print('')
    # del filtered_data
    del raw_data
    # quit()