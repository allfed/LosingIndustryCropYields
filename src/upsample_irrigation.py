'''''
This code  upsamples a csv of irrigation generated by the import code. 

Specifically, it puts it in the form (total reliant)/(crop area),
where if crop area is zero, it is -9 (no data).

Morgan Rivers
morgan@allfed.info
7/24/21
'''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from src import params  # get file location and varname parameters for data import
from src.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
import rasterio
import utilities

# import datetime

# import gc
# import os, psutil
# from pympler import summary
# from pympler import muppy

import resource
from sys import platform
if platform == "linux" or platform == "linux2":
	#this is to ensure Morgan's computer doesn't crash
	import resource
	rsrc = resource.RLIMIT_AS
	resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb

# process = psutil.Process(os.getpid())
# print("mem1: "+str(process.memory_info().rss/1e6))  # in megabytes 
# from pympler.tracker import SummaryTracker
# tracker = SummaryTracker()

#load the params from the params.ods file into the params object
params.importIfNotAlready()
# params.deleteGlobals()

MAKE_GRID = False

mn_lat=-90
mx_lat=90
mn_lon=-180
mx_lon=180

#5 arcminutes in degrees
five_minute=5/60

pSums={}
nbins=1


start_lat_index=np.floor((90-mx_lat)/five_minute).astype('int')
start_lon_index=np.floor((mn_lon+180)/five_minute).astype('int')
print(start_lon_index)
print(start_lat_index)
# we ignore the last latitude cell
lats = np.linspace(-90, 90 - five_minute, \
				   np.floor(180 / five_minute).astype('int'))
lons = np.linspace(-180, 180 - five_minute, \
				   np.floor(360 / five_minute).astype('int'))

print(nbins*len(lats))
print(nbins*len(lons))
# area_result=np.full((nbins*len(lats),nbins*len(lons)),np.nan)
# reliant_result=np.full((nbins*len(lats),nbins*len(lons)),np.nan)
frac_result=np.full((nbins*len(lats),nbins*len(lons)),np.nan)

lats2d, lons2d = np.meshgrid(lats, lons)
data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel())}
df = pd.DataFrame(data=data)

sizeArray=[len(lats),len(lons)]

lowres_irrigation=pd.read_csv(params.geopandasDataDir + 'Irrigation.csv')

#irrigated area
area = np.array(lowres_irrigation['area'].values).astype('float32')

#electric or diesel reliant area
reliant = np.array(lowres_irrigation['tot_reliant'].values).astype('float32')

#nan where not irrigated, where there's a fraction it's irrigated, and the
#fraction is the estimated probability of that area being reliant on 
#electricity or diesel.
lowres_fraction = np.divide(reliant,area)

#let's make sure the input data is 10 times 5 arc minute resolution
#which means nbins was set to 10 in the Params.ods when the irrigation data were
#imported 
# assert(len(lowres_fraction)==93312)
assert(len(area)==373248)

#now, we get a nice numpy array we can upsample
# arrayWithNoData=np.where(np.bitwise_or(array<0, np.isnan(array)), -9, array)
# flippedarr=np.ravel(np.flipud(np.transpose(lowres_fraction.reshape((4320,2160)))))
# area_2d_lowres=area.reshape((int(2160/5),int(4320/5)))/25
# reliant_2d_lowres=reliant.reshape((int(2160/5),int(4320/5)))/25
frac_2d_lowres=area.reshape((int(2160/5),int(4320/5)))

# print(len(area_2d_lowres))
# print(len(area_2d_lowres[0]))

# area_2d_highres=area_2d_lowres.repeat(5, axis=0).repeat(5, axis=1)
# reliant_2d_highres=reliant_2d_lowres.repeat(5, axis=0).repeat(5, axis=1)
frac_2d_highres=frac_2d_lowres.repeat(5, axis=0).repeat(5, axis=1)

# utilities.create5minASCII(manure,'applied',params.asciiDir+'manure')
# print(len(area_2d_highres))
# print(len(area_2d_highres[0]))
# print(len(area_result))
# print(len(area_result[0]))

# area_result[start_lat_index:start_lat_index+len(area_2d_highres),start_lon_index:start_lon_index+len(area_2d_highres[0])]=area_2d_highres
# reliant_result[start_lat_index:start_lat_index+len(reliant_2d_highres),start_lon_index:start_lon_index+len(reliant_2d_highres[0])]=reliant_2d_highres

frac_result[start_lat_index:start_lat_index+len(frac_2d_highres),start_lon_index:start_lon_index+len(frac_2d_highres[0])]=frac_2d_highres


# df['area']=pd.Series((np.transpose(area_result)).ravel())
# df['tot_reliant']=pd.Series((np.transpose(reliant_result)).ravel())

df['frac_reliant']=pd.Series((np.transpose(frac_result)).ravel())


print('done upsampling')

assert(df['lats'].iloc[-1]>df['lats'].iloc[0])
assert(df['lons'].iloc[-1]>df['lons'].iloc[0])
# quit()
# df.sort_values(by=['lats', 'lons'],inplace=True)
# print('2')
# df = df.reset_index(drop=True)
print('saving')
df.to_csv(params.geopandasDataDir + "FracReliantHighRes.csv")
# time10 = datetime.datetime.now()


# print('time1: '+str((time1-start_time).total_seconds() * 1000))
# print('time2: '+str((time2-time1).total_seconds() * 1000))
# print('time3: '+str((time3-time2).total_seconds() * 1000))
# print('time4: '+str((time4-time3).total_seconds() * 1000))
# print('time5: '+str((time5-time4).total_seconds() * 1000))
# print('time6: '+str((time6-time5).total_seconds() * 1000))
# print('time7: '+str((time7-time6).total_seconds() * 1000))
# print('time8: '+str((time8-time7).total_seconds() * 1000))
# print('time9: '+str((time9-time8).total_seconds() * 1000))
# print('time10: '+str((time10-time9).total_seconds() * 1000))
# params.deleteGlobals()

print('done saving')
# all_objects = muppy.get_objects()
# sum1 = summary.summarize(all_objects)
# summary.print_(sum1)
# print("mem2: "+str(psutil.Process().memory_info().rss/1e6))  # in megabytes 
# gc.collect()
# print("mem3: "+str(psutil.Process().memory_info().rss/1e6))  # in megabytes 
