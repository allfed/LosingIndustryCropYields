'''''
This code imports and downsamples a raster (geotiff) of livestock area
Uses DA (Dasymetric) part of dataset.



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

#load the params from the params.ods file into the params object
params.importIfNotAlready()

# aquastat=pd.read_csv(params.aquastatIrrigationDataLoc,index_col=False)
# aq=aquastat.dropna(how='all').replace('',np.nan)
livestocks = ['Bf','Dk','Gt','Pg','Sh','Ho','Ct']
# we ignore the last latitude cell
lats = np.linspace(-90, 90 - params.latdiff, \
			   np.floor(180 / params.latdiff).astype('int'))
lons = np.linspace(-180, 180 - params.londiff, \
			   np.floor(360 / params.londiff).astype('int'))

lats2d, lons2d = np.meshgrid(lats, lons)

sizeArray=[len(lats),len(lons)]

data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel())}


print('reading livestock count data')
for l in livestocks:

	#total animals per pixel
	ldata=rasterio.open(params.livestockDataLoc+'5_'+l+'_2010_Da.tif')
	lArr=ldata.read(1)

	latbins=np.floor(len(lArr)/len(lats)).astype('int')
	lonbins=np.floor(len(lArr[0])/len(lons)).astype('int')
	lArrResized=lArr[0:latbins*len(lats),0:lonbins*len(lons)]
	print('done reading '+l)

	#make data zero if data < 0.
	lArrResizedZeroed=np.where(lArrResized<0, 0, lArrResized)

	lBinned= utilities.rebinCumulative(lArrResizedZeroed, sizeArray)
	lBinnedReoriented=np.fliplr(np.transpose(lBinned))

	data[l] = pd.Series(lBinnedReoriented.ravel())

df = pd.DataFrame(data=data)
geometry = gpd.points_from_xy(df.lons, df.lats)
gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry)

grid= utilities.makeGrid(gdf)
grid.to_pickle(params.geopandasDataDir + "Livestock.pkl")

title="Cattle, 2010"
label="Heads cattle in 2 degree square Cell"
Plotter.plotMap(grid,'Ct',title,label,'HeadsCattle',True)


quit()
#import the country boundaries, so we can see which country coordinates fall into
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#get groups of countries which match the coordinates for population losing electricity
pointInPolys = sjoin(gdf, world)

def get_value(val_id):
	vals=countryRows[countryRows['Variable Id']==val_id]['Value'].values
	if(len(vals)==0):
		return np.nan
	else:
		return vals[0]*1000

#for each country, 
total_gmiav5 = []
total_gmiav5_gw = []
total_gmiav5_sw = []
gmiav5_reliant_areas_sw=[]
gmiav5_reliant_areas_gw=[]
gmiav5_reliant_areas=[]
gmiav5_all_areas=[]
world['reliant']=np.nan
world['reliant_source']=np.nan
world['reliant_scheme']=np.nan
world['reliant_source_sw']=np.nan
world['reliant_scheme_sw']=np.nan
pointInPolys['tot_reliant']=np.nan
pointInPolys['sw_reliant']=np.nan
pointInPolys['gw_reliant']=np.nan
for countryIndex in set(pointInPolys.index_right.values):
	row=world[world.index==countryIndex]
	code=row['iso_a3'].values[0]
	highRes=pointInPolys[pointInPolys['iso_a3']==code]
	countryRows=aq[aq['Area'].str.startswith(code)]

	spate_area=get_value(spate_irrigation_id)
	lowland_area=get_value(lowland_irrigation_id)
	sprinkler_area=get_value(sprinkler_irrigation_id)
	localized_area=get_value(localized_irrigation_id)
	surface_area=get_value(surface_irrigation_id)
	power_area=get_value(power_irrigation_id)
	surface_water_area=get_value(surface_water_id)
	ground_water_area=get_value(ground_water_id)
	total_irrigated_area=get_value(total_equipped_id)
	
	scheme_total_arr=np.array([surface_area,sprinkler_area,localized_area,spate_area,lowland_area])
	scheme_total_arr=scheme_total_arr[~np.isnan(scheme_total_arr)]
	scheme_total=np.sum(scheme_total_arr)
	#assign total irrigated area to the maximum of total area, scheme area, or power area
	areas=np.array([scheme_total,total_irrigated_area,power_area])
	areas=areas[~np.isnan(areas)]
	#not enough data to estimate value.
	if(np.sum(areas)==0):
		max_area=np.nan
		reliant_source = np.nan
		reliant_source_assumed = np.nan
		reliant_scheme = np.nan
		reliant_scheme_assumed = np.nan
		reliant_source_sw=reliant_source_assumed
		reliant_source_nonsw=reliant_source_assumed
		reliant_scheme_sw=reliant_scheme_assumed
		reliant_scheme_nonsw=reliant_scheme_assumed
	else: #at least one of original areas array is not nan and not zero
		max_area=max(areas)

		[reliant_source,reliant_source_assumed,reliant_source_sw,reliant_source_nonsw] = \
			estimate_source_reliance(power_area,total_irrigated_area,max_area,surface_water_area)

		[reliant_scheme,reliant_scheme_assumed,reliant_scheme_sw,reliant_scheme_nonsw]= \
			estimate_scheme_reliance(scheme_total,total_irrigated_area,localized_area,sprinkler_area,spate_area,lowland_area,surface_area,max_area,surface_water_area)

	gmiav5_all_areas.append(highRes['area'].sum())
	if(np.isnan(reliant_scheme) and np.isnan(reliant_source)):
		reliant = np.nan
	else:
		reliant=1-(1-reliant_scheme_assumed)*(1-reliant_source_assumed)
		reliant_sw=1-(1-reliant_scheme_sw)*(1-reliant_source_sw)
		reliant_nonsw=1-(1-reliant_scheme_nonsw)*(1-reliant_source_nonsw)

		if(~np.isnan(surface_water_area)):
			print('')
			print('reliant'+str(reliant))
			print('estreliant'+str((reliant_sw*surface_water_area+reliant_nonsw*(max_area-surface_water_area))/max_area))
			print('reliant_sw_rat'+str(reliant_sw*surface_water_area/max_area)		)
			print('reliant_nonsw_rat'+str(reliant_nonsw*(max_area-surface_water_area)/max_area))

		# print('')
		# print('code'+str(code))
		# print('reliant'+str(reliant))
		# print('reliant_source'+str(reliant_source))
		# print('reliant_source_sw'+str(reliant_source_sw))
		# print('reliant_source_nonsw'+str(reliant_source_nonsw))
		# print('reliant_scheme_assumed'+str(reliant_scheme_assumed))
		# print('reliant_scheme_sw'+str(reliant_scheme_sw))
		# print('reliant_scheme_nonsw'+str(reliant_scheme_nonsw))
		# print('')
		# print('spate_area'+str(spate_area))
		# print('lowland_area'+str(lowland_area))
		# print('sprinkler_area'+str(sprinkler_area))
		# print('localized_area'+str(localized_area))
		# print('surface_area'+str(surface_area))
		# print('power_area'+str(power_area))
		# print('surface_water_area'+str(surface_water_area))
		# print('ground_water_area'+str(ground_water_area))
		# print('total_irrigated_area'+str(total_irrigated_area))



		world.loc[int(row.index[0]),'reliant']=reliant
		world.loc[int(row.index[0]),'reliant_source']=reliant_source
		world.loc[int(row.index[0]),'reliant_source_sw']=reliant_source_sw
		world.loc[int(row.index[0]),'reliant_source_nonsw']=reliant_source_nonsw
		world.loc[int(row.index[0]),'reliant_scheme']=reliant_scheme
		world.loc[int(row.index[0]),'reliant_scheme_sw']=reliant_scheme_sw
		world.loc[int(row.index[0]),'reliant_scheme_nonsw']=reliant_scheme_nonsw
	

		#assert that no ground water in any country 
		#is more likely than surface water to be reliant
		assert(reliant_scheme_nonsw>reliant_scheme_sw-0.001)

		gmiav5_reliant_area=highRes['area'].sum()*reliant
		gmiav5_reliant_area_sw=highRes['surfacewaterArea'].sum()*reliant_sw
		gmiav5_reliant_area_gw=highRes['groundwaterArea'].sum()*reliant_nonsw

		for i, row in pointInPolys[pointInPolys['iso_a3'] == code].iterrows():
			pointInPolys.loc[i,'tot_reliant']=row['area']*reliant
			pointInPolys.loc[i,'sw_reliant']=row['surfacewaterArea']*reliant_sw
			pointInPolys.loc[i,'gw_reliant']=row['groundwaterArea']*reliant_nonsw

		gmiav5_reliant_areas.append(gmiav5_reliant_area)
		gmiav5_reliant_areas_sw.append(gmiav5_reliant_area_sw)
		gmiav5_reliant_areas_gw.append(gmiav5_reliant_area_gw)
		total_gmiav5.append(highRes['area'].sum())
		total_gmiav5_sw.append(highRes['surfacewaterArea'].sum())
		total_gmiav5_gw.append(highRes['groundwaterArea'].sum())


print('estimated fraction irrigation where data available:')
print(np.sum(total_gmiav5)/np.sum(gmiav5_all_areas))
print('estimated year ~2010 reliant irrigation (hectares) where data available:')
print(np.sum(gmiav5_reliant_areas))
print('fraction of global irrigation that is reliant where data available:')
print(np.sum(gmiav5_reliant_areas)/np.sum(total_gmiav5))
print('guess for fraction global surface water irrigation that is reliant:')
print(np.sum(gmiav5_reliant_areas_sw)/np.sum(total_gmiav5_sw))
print('guess for fraction global ground water irrigation that is reliant:')
print(np.sum(gmiav5_reliant_areas_gw)/np.sum(total_gmiav5_gw))

world.plot(column='reliant',
	missing_kwds={
		"color": "lightgrey",
		"edgecolor": "red",
		"hatch": "///",
		"label": "Missing values",
	},
	legend=True,
	legend_kwds={
		'label': "Fraction Reliant",
		'orientation': "horizontal"
	}
)
plt.title('Fraction Overall Reliant on Electricity or Diesel')
plt.show()

world.plot(column='reliant_scheme',
	missing_kwds={
		"color": "lightgrey",
		"edgecolor": "red",
		"hatch": "///",
		"label": "Missing values",
	},
	legend=True,
	legend_kwds={
		'label': "Fraction Reliant",
		'orientation': "horizontal"
	}
)
plt.title('Fraction Scheme Sprinklers or Drip Irrigation')
plt.show()



# world.plot(column='reliant_scheme_sw',
# 	missing_kwds={
# 		"color": "lightgrey",
# 		"edgecolor": "red",
# 		"hatch": "///",
# 		"label": "Missing values",
# 	},
# 	legend=True,
# 	legend_kwds={
# 		'label': "Fraction Reliant",
# 		'orientation': "horizontal"
# 	}
# )

# plt.title('Fraction Surface Water Irrigated Scheme that is Reliant on Electricity or Diesel')
# plt.show()

# world.plot(column='reliant_scheme_nonsw',
# 	missing_kwds={
# 		"color": "lightgrey",
# 		"edgecolor": "red",
# 		"hatch": "///",
# 		"label": "Missing values",
# 	},
# 	legend=True,
# 	legend_kwds={
# 		'label': "Fraction Reliant",
# 		'orientation': "horizontal"
# 	}
# )

# plt.title('Fraction Not Surface Water Irrigation Scheme Reliant on Electricity or Diesel')
# plt.show()


# world.plot(column='reliant_source_sw',
# 	missing_kwds={
# 		"color": "lightgrey",
# 		"edgecolor": "red",
# 		"hatch": "///",
# 		"label": "Missing values",
# 	},
# 	legend=True,
# 	legend_kwds={
# 		'label': "Fraction Reliant",
# 		'orientation': "horizontal"
# 	}
# )

# plt.title('Fraction Surface Water Pumped')
# plt.show()

# world.plot(column='reliant_source_nonsw',
# 	missing_kwds={
# 		"color": "lightgrey",
# 		"edgecolor": "red",
# 		"hatch": "///",
# 		"label": "Missing values",
# 	},
# 	legend=True,
# 	legend_kwds={
# 		'label': "Fraction Reliant",
# 		'orientation': "horizontal"
# 	}
# )

# plt.title('Fraction Non Surface Water Pumped')
# plt.show()



world.plot(column='reliant_source',
	missing_kwds={
		"color": "lightgrey",
		"edgecolor": "red",
		"hatch": "///",
		"label": "Missing values",
	},
	legend=True,
	legend_kwds={
		'label': "Fraction Reliant",
		'orientation': "horizontal"
	}
)
plt.title('Fraction Source Reliant on Pumps')
plt.show()


# now overlay percentage ground and surface water dependent on electricity existing irrigation area.

grid= utilities.makeGrid(pointInPolys)

grid.to_pickle(params.geopandasDataDir + "Irrigation.pkl")

plotGrowArea=True


title="Irrigation Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'area',title,label,'IrrigationGwArea',plotGrowArea)


title="Surface Water Area Irrigation Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'surfacewaterArea',title,label,'IrrigationGwArea',plotGrowArea)

title="Ground Water Area Irrigation Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'groundwaterArea',title,label,'IrrigationSwArea',plotGrowArea)


print("total irrigated area: "+str(grid['area'].sum()))
print("ground water area: "+str(grid['groundwaterArea'].sum()))
print("ground water area as fraction: "+str(grid['groundwaterArea'].sum()/grid['area'].sum()))
print("surface water area: "+str(grid['surfacewaterArea'].sum()))
print("surface water area as fraction: "+str(grid['surfacewaterArea'].sum()/grid['area'].sum()))

# print(grid)
# print(grid.columns)
# print(grid['tot_reliant'])
# print(grid['sw_reliant'])
# print(grid['gw_reliant'])
title="Total Irrigation reliant Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'tot_reliant',title,label,'ReliantIrrigationArea',plotGrowArea)
title="Total Surface Water reliant Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'sw_reliant',title,label,'ReliantIrrigationSwArea',plotGrowArea)
title="Total Ground Water reliant Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'gw_reliant',title,label,'ReliantIrrigationGwArea',plotGrowArea)