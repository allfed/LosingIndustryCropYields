'''

This module performs all the operations necessary to estimate outdoor growth during nuclear winter.

'''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

from src import params
from src import utilities
from src.plotter import Plotter

import csv
from scipy import interpolate
import numpy as np


class OutdoorGrowth:

	def __init__(self):
		params.importIfNotAlready()

	#function saves a CSV of land area of each cell, with temperature, for 
	#each month in the simulation	
	def saveTempCSV(self,ts,growArea):
		with open(params.tempCSVloc, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			allcols=[]
			allcols.append('index')
			allcols.append('arable area (Ha)')
			allcols.append(params.allMonths)
			writer.writerow(allcols)

			for index,row in ts.iterrows():
				allcols=[]
				allcols.append(index)
				area = growArea.iloc[index]['growArea']
				allcols.append(area)
				for m in params.allMonths:
					temp = row[m]
					allcols.append(temp-273.15)
				writer.writerow(allcols)

	#function saves a CSV of land area of each cell, with temperature, for 
	#each month in the simulation	
	def saveLandSunHumRainCSV(self,ts,humidity,sun,rain,growArea):
		with open(params.temphumsunrainCSVloc, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			allcols=[]
			allcols.append('index')
			allcols.append('arable area (Ha)')
			for m in params.allMonths:
				allcols.append(m+" Temp (C)")
				allcols.append(m+" Hum (%)")
				allcols.append(m+" Rain (mm)")
				allcols.append(m+" Sun (W/m^2)")
			writer.writerow(allcols)

			for index,row in ts.iterrows():
				allcols=[]
				allcols.append(index)
				area = growArea.iloc[index]['growArea']
				allcols.append(area)
				for m in params.allMonths:
					temp = row[m]
					h = humidity.iloc[index][m]
					r = rain.iloc[index][m]
					s = sun.iloc[index][m]
					allcols.append(temp-273.15)
					allcols.append(h)
					allcols.append(r * params.rain_mps_to_mm)
					allcols.append(s)
					
				writer.writerow(allcols)

	#function saves a CSV of percent land area, percent water area, and 
	#average windspeed in m/s of each cell	
	def saveWindCSV(self,u,v):
		with open(params.windCSVloc, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			allcols=[]
			allcols.append('index')
			allcols.append('latitude')
			allcols.append('longitude')
			for m in params.allMonths:
				allcols.append(m+" zonal windspeed (along latitude) (m/s)")
				allcols.append(m+" meridional windspeed (along longitude) (m/s)")
			writer.writerow(allcols)

			for index,row in v.iterrows():
				allcols=[]
				allcols.append(index)
				allcols.append(row['lats'])
				allcols.append(row['lons'])

				for m in params.allMonths:
					speed = row[m]
					allcols.append(speed)
				writer.writerow(allcols)
				writer.writerow(allcols)


	#function returns the temperature coefficient interpolated function
	#see params.ods for documentation
	def calcTempCoeffFun(self,crop):
		xvals=[params.Tbase[crop], params.Topt1[crop], params.Topt2[crop],
			   params.Tfp[crop]]
		yvals=[0,1,1,0]
		f = interpolate.interp1d(xvals, yvals, kind='linear', fill_value=0, \
			bounds_error=False)
		return f

	#function returns the rain coefficient interpolated function
	#see params.ods for documentation
	def calcRainCoeffFun(self,crop):
		xvals=[params.Rlow[crop], params.Rpeak[crop], params.Rhigh[crop]]
		yvals=[params.RlowCoeff[crop], params.RpeakCoeff[crop], \
			   params.RhighCoeff[crop]]
		f = interpolate.interp1d(xvals, yvals, kind='linear', fill_value=0,\
			bounds_error=False)
		return f


	#return an array of consecutive months to be used for growing, based on 
	#length of growing season in months
	def getCellGrowMonths(self,temps,nMonths):

		#add nMonths padded to the beginning
		prefix=temps[0:nMonths]
		wrappedtemps=temps+prefix

		#loop through to find maximum set of nMonth months
		circwindow=[]
		for i in range(0,len(wrappedtemps)-nMonths):
			part=wrappedtemps[i:i+nMonths]
			circwindow.append(np.mean(part))

		maxindex=circwindow.index(max(circwindow))
		return params.allMonths[maxindex:maxindex + nMonths]

		for index,row in ts.iterrows():
			allcols=[]
			allcols.append(index)
			area = growArea.iloc[index]['growArea']
			allcols.append(area)
			for m in allMonths:
				temp = row[m]
				allcols.append(temp-273.15)

	# returns crop yield in kg for a cell in the model for the growing months
	def getCellYield(self,crop,raincoeffs,tempcoeffs,growMonths,area):
		cellyield={}
		for month in params.allMonths:
			if(month in growMonths):
				# assumption: each month during growing season represents 
				#equal fraction of total ideal yield
				fractionGrowth= 1 / params.growDuration[crop]
				idealkgPerHa= params.idealGrowth[crop] * fractionGrowth
				estimatedkgPerHa=idealkgPerHa*tempcoeffs[month]*\
					raincoeffs[month]
				estimatedkg=estimatedkgPerHa*area
				cellyield[month]=estimatedkg
			else:
				cellyield[month]=0
		return cellyield

	#Run through each latitude and longitude, and calculate yearly production 
	#in ktons for each crop based on full utilization of cropland for each 
	#crop and growing season months with highest average temperatures in each 
	#cell
	def estimateYields(self,tempData,rainData,growArea,show):
		allyields={}
		for crop in params.allCrops:
			yields=[] #yields for all cells

			tempfun=self.calcTempCoeffFun(crop)
			rainfun=self.calcRainCoeffFun(crop)

			#iterate over temperature data array
			#all months of surface temperature, in units K  
			for index,temprow in tempData.iterrows():

				#cropland area, ha
				area = growArea.iloc[index]['growArea']

				#average rainfall all months, m/s
				rainrow = rainData.iloc[index]

				temps=[]
				tempcoeffs={}
				raincoeffs={}
				for m in params.allMonths:

					rain = rainrow[m]
					#convert m/s to mm rain (assume rain is per year)
					raincoeffs[m]=rainfun(rain * params.rain_mps_to_mm)

					#convert K to degrees C
					temp = temprow[m]-273.15
					tempcoeffs[m]=tempfun(temp)
					temps.append(temp)

				growMonths=self.getCellGrowMonths(temps, \
												  params.growDuration[crop])

				#gets estimated yield in kg for each month for cell
				cellyields=self.getCellYield(crop,raincoeffs,tempcoeffs,\
					growMonths,area)

				cellyields['lats']=temprow['lats']
				cellyields['lons']=temprow['lons']

				yields.append(cellyields)

			#saves the yields, and returns them as a geopandas object
			yieldsgdf = utilities.saveDictasgeopandas(crop + 'Yield', yields)

			self.displayYields(crop,yieldsgdf,show)

			allyields[crop]=yieldsgdf

		return allyields

	#display the yields for each month
	def displayYields(self,crop,yieldsgdf,show):
		for month in params.allMonths:
			print('Month '+month+' yields: '+str(yieldsgdf[month].sum()))

			title = 'Yield for '+crop+' Month '+month+' each Cell'
			label = 'Yields (kg)'
			fn=crop+"Yield"+month
			Plotter.plotMap(yieldsgdf,month,title,label,fn,show)


	#prints nutrition results summary
	def estimateNutrition(crop,yieldsgdf):
		for crop in params.allCrops:
			print('')
			print('Crop: '+crop)
			for month in params.allMonths:
				y=yieldsgdf[crop][month].sum()
				print('  Month: '+month)
				print('    Yield (kg): '+str(y))
				print('    Calories (kCals): ' + str(y *
													 params.kCalperkg[crop]))
				print('    Protein (g): ' + str(y * params.fracProtein[crop]))
				print('    Fat (g): ' + str(y * params.fracFat[crop]))
				print('    Carbs: ' + str(y * params.fracCarbs[crop]))
			print('')
			print('===================')