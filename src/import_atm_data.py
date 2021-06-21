'''
This code imports the netcdf files from Toon group atmosphere dataset. The 
dataset has the nuclear event starting mid-may year 5. The model uses a 
nuclear winter scenario involving 150 Teragrams of soot from burning of cities.

To access the raw data, you need access to the ALLFED account in 
app.globus.com.

Morgan Rivers
morgan@allfed.info
6/6/21
'''

from src import utilities
from src import params  # get file location and varname parameters for data import
import numpy as np
import netCDF4 as nc

# if you'd like to save as geotiff format, you can uncomment block below
# import rasterio
# from rasterio.profiles import DefaultGTiffProfile
# #see https://rasterio.readthedocs.io/en/latest/topics/profiles.html
# #also https://stackoverflow.com/questions/58393244/how-to-export-a-rasterio-window-to-an-image-without-geo-reference-information
# def saveasgeotiff(name,data):
# 	with rasterio.open(\
# 	name+'.tif',\
# 	'w',\
# 	driver='GTiff',\
# 	height=data.shape[0],\
# 	width=data.shape[1],\
# 	count=1,\
# 	dtype=data.dtype,\
# 	crs='+proj=latlong',\
# 	transform=rasterio.Affine(1, 0, 0, 0, 1, 0), #identity affine transform\ 
# 	) as dst:
# 		dst.write(data.astype(rasterio.float32), 1)

#load the params from the params.ods file into the params object
params.importIfNotAlready()

#convert each 2d data array into a geopandas object, and save the object.
lats=np.array([])
lons=np.array([])
allMonths= params.allMonths
print('params to actually import:')
print(params.atmVarNames)
print('')
for vn in params.atmVarNames:
	gridAllMonths=[]
	for month in allMonths:
		ds = nc.Dataset(params.rawAtmDataLoc + month + '.nc')
		# uncomment below to see all the names of the different nc variables (
		#which are specified by latitude and longitude)

		if(month== params.allMonths[0]):
			if(vn== params.atmVarNames[0]):
				print('All possible params:')
				print('###')
				for i in ds.variables.items():
					if(i[1].ndim>2):
						# uncomment section below to see some interesting 
						#variables for this project. Some of them are used.
						print(i[0])
						print(i[1].units)
						print(i[1].long_name)
						print('###')
			lats=np.array(ds['lat'][:-1])
			lons=np.array(ds['lon'])
			lons[lons>=180]=lons[lons>=180]-360
			print('')
			print('Loading '+vn)
		if(np.size(ds[vn][-1])== 13824):
			arr=ds[vn][-1] 
		else: 
			#the variable is layered, take the ground level layer
			arr=ds[vn][0][-1]
		#the last index (-1) appears to be where the gridded data is stored
		grid=np.transpose(arr)
		#remove the latitude starting at 90, as it makes no sense to start a 
		#grid cell at the bottom of the earth
		croppedGrid=grid[:,:-1]

		gridAllMonths.append(croppedGrid)

	latdiff=lats[1]-lats[0]
	londiff=lons[1]-lons[0]
	assert(np.round(latdiff,12) == np.round(float(params.latdiff), 12))
	assert(np.round(londiff,12) == np.round(float(params.londiff), 12))

	print('Saving '+vn)
	#save the imported nuclear winter data in geop andas format
	df= utilities.saveasgeopandas(vn, allMonths, gridAllMonths, lats, lons)
	print('Done Saving '+vn)
