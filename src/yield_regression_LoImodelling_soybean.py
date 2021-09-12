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
from src import stat_ut
from src.outdoor_growth import OutdoorGrowth
import pandas as pd
import geopandas as gpd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for soybean

soyb_yield=pd.read_pickle(params.geopandasDataDir + 'SOYBCropYield.pkl')


#display first 5 rows of soybean yield dataset
soyb_yield.head()

#select all rows from soyb_yield for which the column growArea has a value greater than zero
soyb_nozero=soyb_yield.loc[soyb_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
soyb_kgha=soyb_nozero['yieldPerArea']
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
smean=soyb_kgha.mean()
smeadian=soyb_kgha.median()
ssd=soyb_kgha.std()
svar=soyb_kgha.var()
#logarithmize the values
soyb_kgha_log=np.log(soyb_kgha)

#plot soybean yield distribution in a histogram
plt.hist(soyb_kgha, bins=[1,50, 100, 175, 250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500])
plt.title('Soybean yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yieldPerArea
plt.hist(soyb_kgha_log, bins=[0,1,2,3,4,5,6,7,8,9,10,11])

#test if area without zeros aligns with FAOSTAT harvested area
soyb_area_ha = sum(soyb_nozero['growArea'])
print(soyb_area_ha)
#65436615.57129341
#102767896 FAOSTAT data for 2010



'''
Fitting of distributions to the data and comparing the fit
'''

pdf_list = []
dist_list = []
param_dict ={"Values":[]}
x = np.linspace(0.01,
                4500, 100)


#Lognormal distribution
dist_list.append('lognorm')
#fit distribution to soyb yield data to get values for the parameters
param1 = stats.lognorm.fit(soyb_kgha)
#param_list.append(param1)
param_dict["Values"].append(param1)
print(param1)
#use the parameters to calculate values for the probability density function 
#(pdf) of the distribution
pdf_fitted = stats.lognorm.pdf(x, *param1)
#calculate the logarithmized pdf to calculate statistical values for the fit
pdf_fitted_log = stats.lognorm.logpdf(soyb_kgha, *param1)
pdf_list.append(pdf_fitted_log)
#plot the histogram of the yield data and the curve of the lognorm pdf
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.legend()
plt.show()


#Exponential distribution
dist_list.append('exponential')
#get parameters
param2 = stats.expon.fit(soyb_kgha)
param_dict["Values"].append(param2)
print(param2)
#calculate pdf
pdf_fitted2 = stats.expon.pdf(x, *param2)
#calculate log pdf
pdf_fitted_log2 = stats.expon.logpdf(soyb_kgha, *param2)
pdf_list.append(pdf_fitted_log2)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.legend()
plt.show()


#Weibull distribution
dist_list.append('weibull')
#get parameters
param3 = stats.weibull_min.fit(soyb_kgha)
#param_list.append(param3)
param_dict["Values"].append(param3)
print(param3)
#calculate pdf
pdf_fitted3 = stats.weibull_min.pdf(x, *param3)
#calculate log pdf
pdf_fitted_log3 = stats.weibull_min.logpdf(soyb_kgha, *param3)
pdf_list.append(pdf_fitted_log3)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.legend()
plt.show()


#Gamma distribution
dist_list.append('gamma')
#get parameters
param4 = stats.gamma.fit(soyb_kgha)
#param_list.append(param4)
param_dict["Values"].append(param4)
print(param4)
#calculate pdf
pdf_fitted4 = stats.gamma.pdf(x, *param4)
#calculate log pdf
pdf_fitted_log4 = stats.gamma.logpdf(soyb_kgha, *param4)
pdf_list.append(pdf_fitted_log4)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted4, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()

#Exponential Weibull distribution
dist_list.append('exponential weibull')
#get parameters
param5 = stats.exponweib.fit(soyb_kgha)
#param_list.append(param5)
param_dict["Values"].append(param5)
print(param5)
#calculate pdf
pdf_fitted5 = stats.exponweib.pdf(x, *param5)
#calculate log pdf
pdf_fitted_log5 = stats.exponweib.logpdf(soyb_kgha, *param5)
pdf_list.append(pdf_fitted_log5)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted5, lw=2, label="Fitted exponential weibull distribution")
plt.legend()
plt.show()

#one in all plot
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.plot(x, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.plot(x, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.plot(x, pdf_fitted4, lw=2, label="Fitted Gamma distribution")
plt.plot(x, pdf_fitted5, lw=2, label="Fitted exponential weibull distribution")
plt.legend()
plt.show()


#calculate loglik, AIC & BIC for each distribution
stat_ut.stat_overview(dist_list, pdf_list, param_dict)
#best fit so far: exponential weibull

'''
Load factor data and extract zeros
'''
#load explaining variables
pesticides=pd.read_pickle(params.geopandasDataDir + 'SoybeanPesticidesByCrop.pkl')
fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl')
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')
tillage=pd.read_pickle(params.geopandasDataDir + 'Tillage.pkl')

#print the same row of each dataset to confirm that lats and lons align
print(pesticides.loc[6040])
print(fertilizer.loc[6040])
print(irrigation.loc[6040])
print(tillage.loc[6040])
print(soyb_yield.loc[6040])

#store the relevant columns of dependent and explaining variables in a dictonary
data = {"lat": soyb_yield.loc[:,'lats'],
		"lon": soyb_yield.loc[:,'lons'],
		"area": soyb_yield.loc[:,'growArea'],
        "yield": soyb_yield.loc[:,'yieldPerArea'],
		"n_fertilizer": fertilizer.loc[:,'n'],
		"p_fertilizer": fertilizer.loc[:,'p'],
        "pesticides_L": pesticides.loc[:,'total_L'],
        "pesticides_H": pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'maiz_mech'],
        "non-mechanized": tillage.loc[:,'maiz_non_mech'],
        "irrigation": irrigation.loc[:,'area']
		}

#compile the data from the dictonary in a pandas dataframe
dsoyb = pd.DataFrame(data=data)
#select all rows from dsoyb for which the column area has a value greater than zero
dsoyb_nozero=dsoyb.loc[dsoyb['area'] > 0]

'''
#subplot for all histograms
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(maize_yield['yieldPerArea'], bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 0].hist(maize_kgha, bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[0, 1].hist(maize_kgha_area, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 1].hist(maize_kgha_yield, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])








fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl')
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')

print(fertilizer.columns)
print(fertilizer.head())
# print(irrigation.columns)
# print(fertilizer.columns)
outdoorGrowth=OutdoorGrowth()
outdoorGrowth.correctForFertilizerAndIrrigation(maize_yield,fertilizer,irrigation)
'''