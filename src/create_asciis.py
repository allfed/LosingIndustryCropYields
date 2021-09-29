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

create_fertilizer = input('Would you like to create fertilizer? (enter y/n): \n').lower()
if create_fertilizer.startswith('y'):
	print('Creating Fertilizer')
	fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv')
	utilities.create5minASCII(fertilizer,'n',params.asciiDir+'fertilizer')

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

create_irrigation = input('Would you like to create reliant irrigation? (enter y/n): \n').lower()
if create_irrigation.startswith('y'):
	print('Creating irrigation')

	irrigation=pd.read_csv(params.geopandasDataDir + 'FracReliantHighRes.csv')
	utilities.create5minASCII(irrigation,'frac_reliant',params.asciiDir+'irrigationArea')

create_irrigation_total = input('Would you like to create total irrigation? (enter y/n): \n').lower()
if create_irrigation_total.startswith('y'):
	print('Creating irrigation total')

	irrigation_total=pd.read_csv(params.geopandasDataDir + 'IrrigationHighRes.csv')
	utilities.create5minASCII(irrigation_total,'area',params.asciiDir+'TotIrrigationAreaHighRes')

create_cropland_area = input('Would you like to create total cropland? (enter y/n): \n').lower()
if create_cropland_area.startswith('y'):
	print('Creating crop area')

	cropland=pd.read_csv(params.geopandasDataDir + 'IrrigationHighRes.csv')
	utilities.create5minASCII(cropland,'area',params.asciiDir+'croplandArea')

