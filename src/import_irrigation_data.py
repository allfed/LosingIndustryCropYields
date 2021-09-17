'''''
This code imports and downsamples a raster (geotiff) of crop irrigation area
 from gmiav5 and national level government census data from aquastat.
 Imported data is estimated for 2010 from gmiav5, and usually mid-2010s for 
 aquastat.

see
https://essd.copernicus.org/articles/12/3545/2020/essd-12-3545-2020.pdf
"We prepare datafor the model based on the 2009–2011 average of the cropproduction  statistics"

Output of Import: all units are per cell in hectares
	'area' column: total irrigated area 
	'groundwaterArea' column: total irrigated area using groundwater
	'surfacewaterArea' column: total irrigated area using surfacewater
	'sw_reliant' column: total irrigated area reliant on diesel or electricity 
		using surfacewater
	'gw_reliant' column: total irrigated area reliant on diesel or electricity 
		using groundwater
	'tot_reliant' column: total irrigated area reliant on diesel or electricity


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
import resource
from sys import platform
if platform == "linux" or platform == "linux2":
	#this is to ensure Morgan's computer doesn't crash
	import resource
	rsrc = resource.RLIMIT_AS
	resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb

#load the params from the params.ods file into the params object
params.importIfNotAlready()

aquastat=pd.read_csv(params.aquastatIrrigationDataLoc,index_col=False)
aq=aquastat.dropna(how='all').replace('',np.nan)
surface_irrigation_id=4308
sprinkler_irrigation_id=4309
localized_irrigation_id=4310
equipped_area_id=4311
actually_irrigated=4461
spate_irrigation_id=4316
lowland_irrigation_id=4312
power_irrigation_id=4326
total_equipped_id=4313
flood_recession_id=6043
ground_water_id=4320
surface_water_id=4321

#total area
areadata=rasterio.open(params.irrigationDataLoc+'gmia_v5_aei_ha.asc')
#hectares
gwdata=rasterio.open(params.irrigationDataLoc+'gmia_v5_aeigw_pct_aei.asc')
#surface water
swdata=rasterio.open(params.irrigationDataLoc+'gmia_v5_aeisw_pct_aei.asc')

print('reading irrigation area data')
areaArr=areadata.read(1)
gwArr=gwdata.read(1)
swArr=swdata.read(1)
print('done reading')

# we ignore the last latitude cell
lats = np.linspace(-90, 90 - params.latdiff, \
			   np.floor(180 / params.latdiff).astype('int'))
lons = np.linspace(-180, 180 - params.londiff, \
			   np.floor(360 / params.londiff).astype('int'))

latbins=np.floor(len(areaArr)/len(lats)).astype('int')
lonbins=np.floor(len(areaArr[0])/len(lons)).astype('int')

areaArrResized=areaArr[0:latbins*len(lats),0:lonbins*len(lons)]
gwArrResized=gwArr[0:latbins*len(lats),0:lonbins*len(lons)]
swArrResized=swArr[0:latbins*len(lats),0:lonbins*len(lons)]
sizeArray=[len(lats),len(lons)]

#convert percent to fraction, make it zero if data < 0.
swArrResizedFraction=np.where(swArrResized<0, 0, swArrResized)/100
gwArrResizedFraction=np.where(gwArrResized<0, 0, gwArrResized)/100
swArrResizedFiltered=np.multiply(swArrResizedFraction,areaArrResized)
gwArrResizedFiltered=np.multiply(gwArrResizedFraction,areaArrResized)

areaBinned= utilities.rebinCumulative(areaArrResized, sizeArray)
areaBinnedReoriented=np.fliplr(np.transpose(areaBinned))
swBinned= utilities.rebinCumulative(swArrResizedFiltered, sizeArray)
swBinnedReoriented=np.fliplr(np.transpose(swBinned))
gwBinned = utilities.rebinCumulative(gwArrResizedFiltered, sizeArray)
gwBinnedReoriented=np.fliplr(np.transpose(gwBinned))

lats2d, lons2d = np.meshgrid(lats, lons)

data = {"lats": pd.Series(lats2d.ravel()),
		"lons": pd.Series(lons2d.ravel()),
		"area": pd.Series(areaBinnedReoriented.ravel()),
		"surfacewaterArea": pd.Series(swBinnedReoriented.ravel()),
		"groundwaterArea": pd.Series(gwBinnedReoriented.ravel())
		}

df = pd.DataFrame(data=data)
geometry_plot = gpd.points_from_xy(df.lons, df.lats)
geometry_sjoin = gpd.points_from_xy(df.lons+params.londiff/2., df.lats+params.latdiff/2.)
gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry_sjoin)
gdf['geometry_plot']=geometry_plot
#import the country boundaries, so we can see which country coordinates fall into
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#get groups of countries which match the coordinates for population losing electricity
pointInPolys = sjoin(gdf, world)

print(pointInPolys.columns)
print(pointInPolys.head())
def get_value(val_id):
	vals=countryRows[countryRows['Variable Id']==val_id]['Value'].values
	if(len(vals)==0):
		return np.nan
	else:
		return vals[0]*1000

def estimate_reliant_scheme_sw(reliant_scheme,reliant_scheme_area,sw_area,max_area,spate_area,lowland_area,surface_area,localized_area,sprinkler_area):
	#assume lowland and spate schemes are allocated first to surface water irrigated regions

	if(np.isnan(spate_area)):
		spate_area_zeroed=0
	else:
		spate_area_zeroed=spate_area
	if(np.isnan(lowland_area)):
		lowland_area_zeroed=0
	else:
		lowland_area_zeroed=lowland_area
	if(np.isnan(sprinkler_area)):
		sprinkler_area_zeroed=0
	else:
		sprinkler_area_zeroed=sprinkler_area
	if(np.isnan(localized_area)):
		localized_area_zeroed=0
	else:
		localized_area_zeroed=localized_area
	if(np.isnan(surface_area)):
		surface_area_zeroed=0
	else:
		surface_area_zeroed=surface_area

	unreliant_sw_area=spate_area_zeroed+lowland_area_zeroed

	#if surface water area is defined, and nonzero
	if(~np.isnan(sw_area) and sw_area>0):
		nonsw_area = max_area - sw_area

		#remove unreliant surface water irrigation area from consideration, and assume remaining sprinkler, localized, and surface irrigation are evenly split between ground and surface water irrigated area
	
		if(nonsw_area == 0):
			reliant_sw_area=reliant_scheme_area
			reliant_nonsw_area=0
			reliant_sw=reliant_scheme
			reliant_nonsw=reliant_scheme
		elif(unreliant_sw_area>sw_area):
			if(reliant_scheme_area<nonsw_area):
				#no surface water area assigned reliant
				#reliant area all assigned to nonsw_area
				reliant_sw_area=0
				reliant_nonsw_area=reliant_scheme_area
				reliant_sw=0
				reliant_nonsw=reliant_nonsw_area/nonsw_area
			else:
				#this case is unlikely and hard to work with, so forget about surface water
				reliant_sw_area=reliant_scheme*sw_area
				reliant_nonsw_area=reliant_scheme*nonsw_area
				reliant_sw = reliant_scheme
				reliant_nonsw = reliant_scheme
		else:#simple case, unreliant_sw_area < sw_area
			possibly_reliant_sw_area=sw_area-unreliant_sw_area
			possibly_reliant_nonsw_area=nonsw_area
			#now, even splitting of each, but will need to multiply out by multiplier to get back to original total fraction
			multiplier=(sw_area+nonsw_area)/(sw_area-unreliant_sw_area+nonsw_area)
			reliant_sw_area=possibly_reliant_sw_area*reliant_scheme*multiplier
			reliant_nonsw_area=possibly_reliant_nonsw_area*reliant_scheme*multiplier
			reliant_sw=reliant_sw_area/sw_area
			reliant_nonsw=reliant_nonsw_area/nonsw_area

		if(nonsw_area>0):
			assert(abs(reliant_scheme-(reliant_sw_area/max_area+reliant_nonsw_area/max_area))<0.01)

	else: #surface water isn't defined, so we just use the assumed value
		reliant_nonsw=reliant_scheme
		reliant_sw=reliant_scheme

	return [reliant_sw,reliant_nonsw]

def estimate_source_reliance(power_area,total_irrigated_area,max_area,sw_area):

	if(np.isnan(power_area)):
		reliant_source=np.nan
		reliant_source_assumed=0
		reliant_source_nonsw=reliant_source_assumed
		reliant_source_sw=reliant_source_assumed
	elif(power_area==0):
		reliant_source=0
		reliant_source_assumed=reliant_source
		reliant_source_nonsw=reliant_source_assumed
		reliant_source_sw=reliant_source_assumed

	# we assume the source is reliant 
	# if power_area is greatest area
	elif(
		(np.isnan(total_irrigated_area) or total_irrigated_area<=power_area)
		and 
		scheme_total<=power_area
	):
		reliant_source = 1
		reliant_source_assumed=reliant_source
		reliant_source_nonsw=reliant_source_assumed
		reliant_source_sw=reliant_source_assumed
	else:
		reliant_source = power_area/max_area
		reliant_source_assumed=reliant_source

		#determine fraction of reliance that is likely surface water vs ground water
		#assume pumped electricity area are allocated first to non-surface water regions

		if(~np.isnan(sw_area)):
			nonsw_area=max_area-sw_area
			if(nonsw_area>power_area):
				reliant_source_nonsw=power_area/nonsw_area
				reliant_source_sw=0
			else:#nonsurface water area <= power area 
				reliant_source_nonsw=1
				reliant_source_sw=(power_area-nonsw_area)/(sw_area)
		else: #data unavailable, don't assume difference between ground and surface water
			reliant_source_nonsw=reliant_source_assumed
			reliant_source_sw=reliant_source_assumed

	return [reliant_source,reliant_source_assumed,reliant_source_sw,reliant_source_nonsw]

def estimate_scheme_reliance(scheme_total,total_irrigated_area,localized_area,sprinkler_area,spate_area,lowland_area,surface_area,max_area,surface_water_area):
	#for now, assume total irrigated area is zero if not listed
	if(np.isnan(total_irrigated_area)):
		total_irrigated_area_zeroed=0
	else:
		total_irrigated_area_zeroed=total_irrigated_area
	if(np.isnan(spate_area)):
		spate_area_zeroed=0
	else:
		spate_area_zeroed=spate_area
	if(np.isnan(lowland_area)):
		lowland_area_zeroed=0
	else:
		lowland_area_zeroed=lowland_area
	if(np.isnan(sprinkler_area)):
		sprinkler_area_zeroed=0
	else:
		sprinkler_area_zeroed=sprinkler_area
	if(np.isnan(localized_area)):
		localized_area_zeroed=0
	else:
		localized_area_zeroed=localized_area
	if(np.isnan(surface_area)):
		surface_area_zeroed=0
	else:
		surface_area_zeroed=surface_area
	if(np.isnan(surface_water_area)):
		surface_water_area_zeroed=0
	else:
		surface_water_area_zeroed=surface_water_area
		
	#catch issues with incomplete or nonsense data
	#(ignore aquastat surface and ground water area data, if data is not complete or is erroneous)
	if(
		abs(scheme_total)-1 
		> abs(total_irrigated_area_zeroed) 
		or 
		abs(scheme_total) + 1 
		< abs(total_irrigated_area_zeroed)
		):		

		# in this case we don't have good direct estimate regarding the fraction of the scheme that is reliant. We'll guess all remaining area is electrified if spate or lowland are listed explicitly,
		# as long as neither sprinkler or localized are listed explicitly as zero
		if(localized_area_zeroed+sprinkler_area_zeroed==0):
			if(localized_area==0 or sprinkler_area==0):
				reliant_scheme=np.nan
				reliant_scheme_assumed=0	
				reliant_scheme_nonsw=reliant_scheme_assumed
				reliant_scheme_sw=reliant_scheme_assumed
			elif(#both localized_area and sprinkler area must have been nan
				spate_area_zeroed+lowland_area_zeroed+surface_area_zeroed>max_area/2
				):
				reliant_scheme =\
					(max_area-(spate_area_zeroed+lowland_area_zeroed+surface_area_zeroed))/max_area
				reliant_scheme_assumed=reliant_scheme
				reliant_scheme_area=max_area-(spate_area_zeroed+lowland_area_zeroed+surface_area_zeroed)
				[reliant_scheme_sw,reliant_scheme_nonsw]=estimate_reliant_scheme_sw(reliant_scheme_assumed,reliant_scheme_area,surface_water_area,max_area,spate_area,lowland_area,surface_area,localized_area,sprinkler_area)

			#we don't really know anything about it.
			else:
				reliant_scheme=np.nan
				reliant_scheme_assumed=0
				reliant_scheme_nonsw=reliant_scheme_assumed
				reliant_scheme_sw=reliant_scheme_assumed				
		else:
			reliant_scheme=(localized_area_zeroed+sprinkler_area_zeroed)/max_area
			reliant_scheme_assumed=reliant_scheme
			reliant_scheme_area=localized_area_zeroed+sprinkler_area_zeroed
			[reliant_scheme_sw,reliant_scheme_nonsw]=estimate_reliant_scheme_sw(reliant_scheme_assumed,reliant_scheme_area,surface_water_area,max_area,spate_area,lowland_area,surface_area,localized_area,sprinkler_area)
	#good data with nonzero values, complete irrigation data, surface water greater than spate plus lowland irrigation 
	else:
		reliant_scheme=(localized_area_zeroed+sprinkler_area_zeroed)/max_area
		reliant_scheme_assumed=reliant_scheme
		reliant_scheme_area=localized_area_zeroed+sprinkler_area_zeroed
		[reliant_scheme_sw,reliant_scheme_nonsw]=estimate_reliant_scheme_sw(reliant_scheme_assumed,reliant_scheme_area,surface_water_area,max_area,spate_area,lowland_area,surface_area,localized_area,sprinkler_area)

	return [reliant_scheme,reliant_scheme_assumed,reliant_scheme_sw,reliant_scheme_nonsw]

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
gdf['tot_reliant']=0
gdf['sw_reliant']=0
gdf['gw_reliant']=0
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

		# if(~np.isnan(surface_water_area)):
			# print('')
			# print('reliant'+str(reliant))
			# print('estreliant'+str((reliant_sw*surface_water_area+reliant_nonsw*(max_area-surface_water_area))/max_area))
			# print('reliant_sw_rat'+str(reliant_sw*surface_water_area/max_area)		)
			# print('reliant_nonsw_rat'+str(reliant_nonsw*(max_area-surface_water_area)/max_area))

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
			gdf.loc[i,'tot_reliant']=row['area']*reliant
			gdf.loc[i,'sw_reliant']=row['surfacewaterArea']*reliant_sw
			gdf.loc[i,'gw_reliant']=row['groundwaterArea']*reliant_nonsw

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

title='Fraction Overall Reliant on Electricity or Diesel'
label='Fraction Reliant'
fn='OverallReliantCountries'
Plotter.plotCountryMaps(world,'reliant',title,label,fn,True)

title='Fraction Scheme Sprinklers or Drip Irrigation'
label='Fraction Reliant'
fn='SchemeReliantCountries'
Plotter.plotCountryMaps(world,'reliant_scheme',title,label,fn,True)

title='Fraction Source Reliant on Pumps'
label='Fraction Reliant'
fn='SourceReliantCountries'
Plotter.plotCountryMaps(world,'reliant_source',title,label,fn,True)


gdf['geometry']=gdf['geometry_plot']
# now overlay percentage ground and surface water dependent on electricity existing irrigation area.
grid= utilities.makeGrid(gdf)

grid.to_csv(params.geopandasDataDir + "Irrigation.csv")

plotGrowArea=True


title="Irrigation Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'area',title,label,'IrrigationArea2005',plotGrowArea)


title="Surface Water Area Irrigation Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'surfacewaterArea',title,label,'IrrigationSwArea2005',plotGrowArea)

title="Ground Water Area Irrigation Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'groundwaterArea',title,label,'IrrigationGwArea2005',plotGrowArea)


print("total irrigated area: "+str(grid['area'].sum()))
print("ground water area: "+str(grid['groundwaterArea'].sum()))
print("ground water area as fraction: "+str(grid['groundwaterArea'].sum()/grid['area'].sum()))
print("surface water area: "+str(grid['surfacewaterArea'].sum()))
print("surface water area as fraction: "+str(grid['surfacewaterArea'].sum()/grid['area'].sum()))

title="Total Irrigation reliant Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'tot_reliant',title,label,'ReliantIrrigationArea2005',plotGrowArea)
title="Total Surface Water reliant Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'sw_reliant',title,label,'ReliantIrrigationSwArea2005',plotGrowArea)
title="Total Ground Water reliant Area, 2005"
label="Area (ha)"
Plotter.plotMap(grid,'gw_reliant',title,label,'ReliantIrrigationGwArea2005',plotGrowArea)