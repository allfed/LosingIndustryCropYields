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
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for rice

rice_yield=pd.read_pickle(params.geopandasDataDir + 'RICECropYield.pkl')

#display first 5 rows of rice yield dataset
rice_yield.head()

#select all rows from rice_yield for which the column growArea has a value greater than zero
rice_nozero=rice_yield.loc[rice_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
rice_kgha=rice_nozero['yield_kgPerHa']
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
rmean=rice_kgha.mean()
rmeadian=rice_kgha.median()
rsd=rice_kgha.std()
rvar=rice_kgha.var()
#logarithmize the values
rice_kgha_log=np.log(rice_kgha)

#plot rice yield distribution in a histogram
plt.hist(rice_kgha, bins=50)
plt.title('Rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yield_kgPerHa
plt.hist(rice_kgha_log, bins=50)

#test if area without zeros aligns with FAOSTAT harvested area
rice_area_ha = sum(rice_nozero['growArea'])
print(rice_area_ha)
#160732191.87618524
#160256712 FAOSTAT data for year 2010



'''
Fitting of distributions to the data and comparing the fit
'''

pdf_list = []
dist_list = []
param_dict ={"Values":[]}
x = np.linspace(0.01,
                16000, 100)


#Normal distribution
dist_list.append('norm')
#fit distribution to rice yield data to get values for the parameters
param1 = stats.norm.fit(rice_kgha)
#param_list.append(param1)
param_dict["Values"].append(param1)
print(param1)
#use the parameters to calculate values for the probability density function 
#(pdf) of the distribution
pdf_fitted = stats.norm.pdf(x, *param1)
#calculate the logarithmized pdf to calculate statistical values for the fit
pdf_fitted_log = stats.norm.logpdf(rice_kgha, *param1)
pdf_list.append(pdf_fitted_log)
#plot the histogram of the yield data and the curve of the lognorm pdf
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted, lw=2, label="Fitted Normal distribution")
plt.legend()
plt.show()


#Exponential distribution
dist_list.append('exponential')
#get parameters
param2 = stats.expon.fit(rice_kgha)
param_dict["Values"].append(param2)
print(param2)
#calculate pdf
pdf_fitted2 = stats.expon.pdf(x, *param2)
#calculate log pdf
pdf_fitted_log2 = stats.expon.logpdf(rice_kgha, *param2)
pdf_list.append(pdf_fitted_log2)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.legend()
plt.show()


#Weibull distribution
dist_list.append('weibull')
#get parameters
param3 = stats.weibull_min.fit(rice_kgha)
#param_list.append(param3)
param_dict["Values"].append(param3)
print(param3)
#calculate pdf
pdf_fitted3 = stats.weibull_min.pdf(x, *param3)
#calculate log pdf
pdf_fitted_log3 = stats.weibull_min.logpdf(rice_kgha, *param3)
pdf_list.append(pdf_fitted_log3)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.ylim(top=0.001)
plt.legend()
plt.show()


#Gamma distribution
dist_list.append('gamma')
#get parameters
param4 = stats.gamma.fit(rice_kgha)
#param_list.append(param4)
param_dict["Values"].append(param4)
print(param4)
#calculate pdf
pdf_fitted4 = stats.gamma.pdf(x, *param4)
#calculate log pdf
pdf_fitted_log4 = stats.gamma.logpdf(rice_kgha, *param4)
pdf_list.append(pdf_fitted_log4)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted4, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()

x1 = np.linspace(4,
                11, 100)
#normal distribution on log values
dist_list.append('normal on log')
#get parameters
param5 = stats.norm.fit(rice_kgha_log)
#param_list.append(param5)
param_dict["Values"].append(param5)
print(param5)
#calculate pdf
pdf_fitted5 = stats.norm.pdf(x1, *param5)
#calculate log pdf
pdf_fitted_log5 = stats.norm.logpdf(rice_kgha_log, *param5)
pdf_list.append(pdf_fitted_log5)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha_log, bins=50, density=True)
plt.plot(x1, pdf_fitted5, lw=2, label="Fitted normal distribution on log")
plt.legend()
plt.show()

#one in all plot
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted, lw=2, label="Fitted Normal distribution")
plt.plot(x, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.plot(x, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.plot(x, pdf_fitted4, lw=2, label="Fitted Gamma distribution")
#plt.plot(x, pdf_fitted5, lw=2, label="Fitted Normal distribution on log")
plt.legend()
plt.title('Rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.show()




#calculate loglik, AIC & BIC for each distribution
stat_ut.stat_overview(dist_list, pdf_list, param_dict)
#    Distribution  loglikelihood           AIC           BIC
#4  normal on log   -2413.944533   4837.889065   4866.572783
#3          gamma  -20713.772950  41437.545900  41466.229618
#1    exponential  -20984.385966  41978.771932  42007.455651
#0           norm  -21036.378323  42082.756646  42111.440365
#2        weibull  -26357.915039  52725.830078  52754.513796
#best fit so far: normal on log values by far


john_bic = stat_ut.biccont(pdf_fitted_log5, param5)
print(john_bic)
gamma_bic = stat_ut.biccont(pdf_fitted_log4, param4)
print(gamma_bic)
weibull_bic = stat_ut.biccont(pdf_fitted_log3, param3)
print(weibull_bic)
print(param_dict)

'''
Load factor data and extract zeros
'''
#load explaining variables
pesticides=pd.read_pickle(params.geopandasDataDir + 'RicePesticidesByCrop.pkl')
fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl')
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')
tillage=pd.read_pickle(params.geopandasDataDir + 'Tillage.pkl')

#print the same row of each dataset to confirm that lats and lons align
print(pesticides.loc[6040])
print(fertilizer.loc[6040])
print(irrigation.loc[6040])
print(tillage.loc[6040])
print(rice_yield.loc[6040])

#store the relevant columns of dependent and explaining variables in a dictonary
data = {"lat": rice_yield.loc[:,'lats'],
		"lon": rice_yield.loc[:,'lons'],
		"area": rice_yield.loc[:,'growArea'],
        "yield": rice_yield.loc[:,'yield_kgPerHa'],
		"n_fertilizer": fertilizer.loc[:,'n'],
		"p_fertilizer": fertilizer.loc[:,'p'],
        "pesticides_L": pesticides.loc[:,'total_L'],
        "pesticides_H": pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'maiz_mech'],
        "non-mechanized": tillage.loc[:,'maiz_non_mech'],
        "irrigation": irrigation.loc[:,'area']
		}

#compile the data from the dictonary in a pandas dataframe
drice = pd.DataFrame(data=data)
#select all rows from drice for which the column area has a value greater than zero
drice_nozero=drice.loc[drice['area'] > 0]

'''
#subplot for all histograms
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(rice_yield['yield_kgPerHa'], bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 0].hist(rice_kgha, bins=[1,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[0, 1].hist(rice_kgha_area, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])
axs[1, 1].hist(rice_kgha_yield, bins=[1,50, 100, 175, 250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000])








fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl')
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')

print(fertilizer.columns)
print(fertilizer.head())
# print(irrigation.columns)
# print(fertilizer.columns)
outdoorGrowth=OutdoorGrowth()
outdoorGrowth.correctForFertilizerAndIrrigation(rice_yield,fertilizer,irrigation)
'''