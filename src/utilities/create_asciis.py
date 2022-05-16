'''

Create a bunch of nice ascii files to look at before you run the stats

'''

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from src.utilities import params  # get file location and varname parameters for data import
from src.utilities.plotter import Plotter
import src.utilities.utilities as utilities


from sys import platform
if platform == "linux" or platform == "linux2":
    #this is to ensure Morgan's computer doesn't crash
    import resource
    rsrc = resource.RLIMIT_AS
    resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb


params.importAll()

create_yield = input('Would you like to create Maize Yield? (enter y/n): \n').lower()
if create_yield.startswith('y'):
    print('Creating Maize Yield')
    fertilizer=pd.read_csv(params.geopandasDataDir + 'MAIZCropYieldHighRes.csv')
    utilities.create5minASCII(fertilizer,'yield_kgPerHa',params.asciiDir+'yield_maiz')

create_fertilizer_n = input('Would you like to create fertilizer? (enter y/n): \n').lower()
if create_fertilizer_n.startswith('y'):
    print('Creating Fertilizer')
    fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv')
    utilities.create5minASCII(fertilizer,'n',params.asciiDir+'fertilizer')
    
create_fertilizer_p = input('Would you like to create p fertilizer? (enter y/n): \n').lower()
if create_fertilizer_p.startswith('y'):
    print('Creating P Fertilizer')
    fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv')
    utilities.create5minASCII(fertilizer,'p',params.asciiDir+'fertilizer_p')

create_tillage = input('Would you like to create tillage? (enter y/n): \n').lower()
if create_tillage.startswith('y'):
    print('Creating tillage')
    tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighReswhea.csv')
    utilities.create5minASCII(tillage,'whea_is_mech',params.asciiDir+'tillageWheat')
create_tillage = input('Would you like to create all tillage? (enter y/n): \n').lower()
if create_tillage.startswith('y'):
    print('Creating all tillage')
    tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighResAllCrops.csv')
    utilities.create5minASCII(tillage,'is_mech',params.asciiDir+'tillageAll')

create_pesticides = input('Would you like to create wheat pesticides? (enter y/n): \n').lower()
if create_pesticides.startswith('y'):
    print('Creating pesticides')

    pesticides=pd.read_csv(params.geopandasDataDir + 'WheatTriallateHighRes.csv')
    utilities.create5minASCII(pesticides,'total_H',params.asciiDir+'pesticidesWheat')
    
create_pesticides = input('Would you like to create corn pesticides? (enter y/n): \n').lower()
if create_pesticides.startswith('y'):
    print('Creating pesticides')

    pesticides=pd.read_csv(params.geopandasDataDir + 'CornPesticidesHighRes.csv')
    utilities.create5minASCII(pesticides,'total_H',params.asciiDir+'pesticidesCorn')

create_manure = input('Would you like to create manure? (enter y/n): \n').lower()
if create_manure.startswith('y'):
    print('Creating manure')

    manure=pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv')
    utilities.create5minASCII(manure,'applied',params.asciiDir+'manure')

create_aez = input('Would you like to create aez? (enter y/n): \n').lower()
if create_aez.startswith('y'):
    print('Creating aez')

    aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')
    utilities.create5minASCII(aez,'soil',params.asciiDir+'aezSoil')

create_irrigation = input('Would you like to create fraction irrigation that is reliant irrigation? (enter y/n): \n').lower()
if create_irrigation.startswith('y'):
    print('Creating irrigation')

    irrigation=pd.read_csv(params.geopandasDataDir + 'FracReliantHighRes.csv')
    utilities.create5minASCII(irrigation,'frac_reliant',params.asciiDir+'irrigationReliantFraction')

create_irrigation_total = input('Would you like to create fraction cell area that is irrigated? (enter y/n): \n').lower()
if create_irrigation_total.startswith('y'):
    print('Creating irrigation total')

    irrigation_frac=pd.read_csv(params.geopandasDataDir + 'FracIrrigationAreaHighRes.csv')
    utilities.create5minASCII(irrigation_frac,'fraction',params.asciiDir+'irrigationAreaFraction')

create_cropland_area = input('Would you like to create fraction cell that is cropland? (enter y/n): \n').lower()
if create_cropland_area.startswith('y'):
    print('Creating crop area')

    cropland=pd.read_csv(params.geopandasDataDir + 'FracCropAreaHighRes.csv')
    utilities.create5minASCII(cropland,'fraction',params.asciiDir+'croplandAreaFraction')


create_cell_area = input('Would you like to create cell area? (enter y/n): \n').lower()
if create_cell_area.startswith('y'):
    print('Creating cell area')

    cell=pd.read_csv(params.geopandasDataDir + 'CellAreaHighRes.csv')
    utilities.create5minASCII(cell,'area',params.asciiDir+'cellAreaFraction')
create_LoI_cy2 = input('Would you like to create maize LoI yield gap change for year 2? (enter y/n): \n').lower()
if create_LoI_cy2.startswith('y'):
    print('Creating LoI y2')

    cell=pd.read_csv(params.geopandasDataDir + 'mzeLOIYieldGapHighRes.csv')
    utilities.create5minASCII(cell,'m_y2_change',params.asciiDir+'LoI_yield_gap_y2')
    
# create_LoI_y2 = input('Would you like to create maize LoI yield gap for year 2? (enter y/n): \n').lower()
# if create_LoI_y2.startswith('y'):
#   print('Creating LoI y2')

#   cell=pd.read_csv(params.geopandasDataDir + 'LoImaizeGapFiltered.csv')
#   utilities.create5minASCII(cell,'m_yield_y2',params.asciiDir+'LoI_yield_gap_y2')
    


yield_gap = input('Would you like to create maize yield gap? (enter y/n): \n').lower()
if yield_gap.startswith('y'):
    print('Creating yield gap')

    cell=pd.read_csv(params.geopandasDataDir + 'mzeYieldGapHighRes.csv')
    utilities.create5minASCII(cell,'gap',params.asciiDir+'1minusYieldGap')
    
        
create_LoI_y2 = input('Would you like to create LoI yield for year 2? (enter y/n): \n').lower()
if create_LoI_y2.startswith('y'):
    print('Creating LoI y2')

    cell=pd.read_csv(params.geopandasDataDir + 'LoIy2Filtered.csv')
    utilities.create5minASCII(cell,'0',params.asciiDir+'LoI_yield_y2')
    
create_LoI_cy2 = input('Would you like to create LoI yield change for year 2? (enter y/n): \n').lower()
if create_LoI_cy2.startswith('y'):
    print('Creating LoI y2')

    cell=pd.read_csv(params.geopandasDataDir + 'LoIy2Filtered.csv')
    utilities.create5minASCII(cell,'1',params.asciiDir+'LoI_yieldchange_y2')

