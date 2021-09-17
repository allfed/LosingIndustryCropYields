'''

File containing the code to explore data and perform a multiple regression
on yield for wheat
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
from src import stat_ut
import pandas as pd
import geopandas as gpd
import scipy
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import inf
#seaborn is just used for plotting, might be removed later
import seaborn as sb
import statsmodels.api as sm
from patsy import dmatrices
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for wheat

wheat_yield=pd.read_csv(params.geopandasDataDir + 'WHEACropYieldHighRes.csv')

#display first 5 rows of wheat yield dataset
wheat_yield.head()

#select all rows from wheat_yield for which the column growArea has a value greater than zero
wheat_nozero=wheat_yield.loc[wheat_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
wheat_kgha=wheat_nozero['yield_kgPerHa']
wheat_kg = wheat_nozero['totalYield']
wweights=wheat_nozero['growArea']/wheat_nozero['growArea'].sum()
wweigh_sum = wweights.sum()
print(wheat_nozero['growArea'].max())
print(wheat_nozero['growArea'].min())
wheat_weighted=wheat_kgha*wweights*wheat_nozero['growArea'].sum()
print(wheat_weighted.mean())
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
wmean=wheat_kgha.mean()
wmedian=wheat_kgha.median()
wsd=wheat_kgha.std()
wvar=wheat_kgha.var()
wmax=wheat_kgha.max()
#calculate the mean with total production and area to check if the computed means align
wmean_total = ( wheat_nozero['totalYield'].sum()) / (wheat_nozero['growArea'].sum())
#the means do not align, probably due to the rebinning process
#calculate weighted mean (by area) of the yield colum
wmean_weighted = round(np.average(wheat_kgha, weights=wheat_nozero['growArea']),2)
#now they align!

#check the datatype of yield_kgPerHa and logarithmize the values
#logging is done to check the histogram and regoression fit of the transformed values
wheat_kgha.dtype
wheat_kgha_log=np.log(wheat_kgha)

#plot wheat yield distribution in a histogram
plt.hist(wheat_kgha, bins=50)
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=15000)
plt.hist(wheat_kg, bins=50)
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg')
plt.ylabel('density')
plt.xlim(right=30000000)
plt.hist(wheat_weighted, bins=100)
plt.title('wheat yield kg/ha')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=300000)

#plot log transformed values of yield_kgPerHa
plt.hist(wheat_kgha_log, bins=50)

#test if area without zeros aligns with FAOSTAT harvested area
wheat_area_ha = sum(wheat_nozero['growArea'])
print(wheat_area_ha)
#164569574.0937798
#164586904	#FAOSTAT area from 2010 for wheat

#subplot for all histograms
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[0].hist(wheat_kgha, bins=50)
axs[1].hist(wheat_kgha_log, bins=50)


#plots show that the raw values are right skewed so we try to fit a lognormal distribution and an exponentail distribution
#on the raw data and a normal distribution on the log transformed data

'''
Fitting of distributions to the data and comparing the fit
'''
#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#initialize a list for density functions, distribution names and estimated parameters
#the lists will later be used to calculate logLik, AIC and BIC of each distribution
#to compare the fits against each other
pdf_listw = []
dist_listw = []
param_dictw ={"Values":[]}
#set xw to bins in the range of the raw data
xw = np.linspace(0.01,
                16500, 100)

###################################################################################
#####Testing of multiple distributions visually and using logLik, AIC and BIC######
###################################################################################

#Lognormal distribution
dist_listw.append('lognorm')
#fit distribution to rice yield data to get values for the parameters
param1 = stats.lognorm.fit(wheat_kgha)
#store the parameters in the initialized dictionary
param_dictw["Values"].append(param1)
print(param1)
#use the parameters to calculate values for the probability density function 
#(pdf) of the distribution
pdf_fitted = stats.lognorm.pdf(xw, *param1)
#calculate the logarithmized pdf to calculate statistical values for the fit
pdf_fitted_log = stats.lognorm.logpdf(wheat_kgha, *param1)
#store the log pdf in the pdf list
pdf_listw.append(pdf_fitted_log)
#plot the histogram of the yield data and the curve of the lognorm pdf
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.legend()
plt.show()


#Exponential distribution
dist_listw.append('exponential')
#get parameters and store them in the dictionary
param2 = stats.expon.fit(wheat_kgha)
param_dictw["Values"].append(param2)
print(param2)
#calculate pdf
pdf_fitted2 = stats.expon.pdf(xw, *param2)
#calculate log pdf and store it in the list
pdf_fitted_log2 = stats.expon.logpdf(wheat_kgha, *param2)
pdf_listw.append(pdf_fitted_log2)
#plot data histogram and pdf curve
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.legend()
plt.show()


#Weibull distribution
dist_listw.append('weibull')
#get parameters and store them in the dictionary
param3 = stats.weibull_min.fit(wheat_kgha)
#param_list.append(param3)
param_dictw["Values"].append(param3)
print(param3)
#calculate pdf
pdf_fitted3 = stats.weibull_min.pdf(xw, *param3)
#calculate log pdf and store it in the list
pdf_fitted_log3 = stats.weibull_min.logpdf(wheat_kgha, *param3)
pdf_listw.append(pdf_fitted_log3)
#plot data histogram and pdf curve
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.ylim(top=0.00030)
plt.legend()
plt.show()

#Normal distribution
dist_listw.append('normal')
#get parameters and store them in the dictionary
param4 = stats.norm.fit(wheat_kgha)
#param_list.append(param4)
param_dictw["Values"].append(param4)
print(param4)
#calculate pdf
pdf_fitted4 = stats.norm.pdf(xw, *param4)
#calculate log pdf and store it in the list
pdf_fitted_log4 = stats.norm.logpdf(wheat_kgha, *param4)
pdf_listw.append(pdf_fitted_log4)
#plot data histogram and pdf curve
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted4, lw=2, label="Fitted normal distribution")
plt.legend()
plt.show()

#Halfnorm distribution
dist_listw.append('halfnormal')
#get parameters and store them in the dictionary
param5 = stats.halfnorm.fit(wheat_kgha)
#param_list.append(param5)
param_dictw["Values"].append(param5)
print(param5)
#calculate pdf
pdf_fitted5 = stats.halfnorm.pdf(xw, *param5)
#calculate log pdf and store it in the list
pdf_fitted_log5 = stats.halfnorm.logpdf(wheat_kgha, *param5)
pdf_listw.append(pdf_fitted_log5)
#plot data histogram and pdf curve
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted5, lw=2, label="Fitted halfnormal distribution")
plt.legend()
plt.show()

#Gamma distribution
dist_listw.append('Gamma')
#get parameters and store them in the dictionary
param6 = stats.gamma.fit(wheat_kgha)
#param_list.append(param6)
param_dictw["Values"].append(param6)
print(param6)
#calculate pdf
pdf_fitted6 = stats.gamma.pdf(xw, *param6)
#calculate log pdf and store it in the list
pdf_fitted_log6 = stats.gamma.logpdf(wheat_kgha, *param6)
pdf_listw.append(pdf_fitted_log6)
#plot data histogram and pdf curve
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted6, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()

#Inverse Gamma distribution
dist_listw.append('Inverse Gamma')
#get parameters and store them in the dictionary
param7 = stats.invgamma.fit(wheat_kgha)
#param_list.append(param5)
param_dictw["Values"].append(param7)
print(param7)
#calculate pdf
pdf_fitted7 = stats.invgamma.pdf(xw, *param7)
#calculate log pdf and store it in the list
pdf_fitted_log7 = stats.invgamma.logpdf(wheat_kgha, *param7)
pdf_listw.append(pdf_fitted_log7)
#plot data histogram and pdf curve
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted7, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.show()

xw1 = np.linspace(4,
                11, 100)
#Normal distribution on log values
dist_listw.append('normal on log')
#get parameters and store them in the dictionary
param8 = stats.norm.fit(wheat_kgha_log)
#param_list.append(param4)
param_dictw["Values"].append(param8)
print(param8)
#calculate pdf
pdf_fitted8 = stats.norm.pdf(xw1, *param8)
#calculate log pdf and store it in the list
pdf_fitted_log8 = stats.norm.logpdf(wheat_kgha_log, *param8)
pdf_listw.append(pdf_fitted_log8)
#plot data histogram and pdf curve
h = plt.hist(wheat_kgha_log, bins=50, density=True)
plt.plot(xw1, pdf_fitted8, lw=2, label="Fitted normal distribution on log")
plt.legend()
plt.title('log wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.show()

#one in all plot
h = plt.hist(wheat_kgha, bins=50, density=True)
plt.plot(xw, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.plot(xw, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.plot(xw, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.plot(xw, pdf_fitted4, lw=2, label="Fitted Normal distribution")
plt.plot(xw, pdf_fitted5, lw=2, label="Fitted Halfnormal distribution")
plt.plot(xw, pdf_fitted6, lw=2, label="Fitted Gamma distribution")
plt.plot(xw, pdf_fitted7, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=20000)
plt.show()


#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_listw, pdf_listw, param_dictw)
'''
      Distribution  loglikelihood           AIC           BIC
7  normal on log  -5.446560e+05  1.089328e+06  1.089416e+06
6  Inverse Gamma  -4.059595e+06  8.119207e+06  8.119295e+06
4     halfnormal  -4.080175e+06  8.160366e+06  8.160455e+06
1    exponential  -4.102465e+06  8.204946e+06  8.205034e+06
3         normal  -4.166714e+06  8.333444e+06  8.333532e+06
0        lognorm  -5.009504e+06  1.001902e+07  1.001911e+07
2        weibull  -5.153811e+06  1.030764e+07  1.030773e+07
5          Gamma           -inf           inf           inf
#best fit so far: normal on log values by far, then Gamma on non-log
'''

'''
Load factor data and extract zeros
'''
pesticides=pd.read_csv(params.geopandasDataDir + 'WheatPesticidesByCropHighRes.csv')
print(pesticides.columns)
print(pesticides.head())
fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv') #kg/m²
print(fertilizer.columns)
print(fertilizer.head())
fertilizer_man=pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv') #kg/km²
print(fertilizer_man.columns)
print(fertilizer_man.head())
#irrigation=pd.read_csv(params.geopandasDataDir + 'IrrigationHighRes.csv')
#print(irrigation.columns)
#print(irrigation.head())
#check on Tillage because Morgan didn't change the allocation of conservation agriculture to mechanized
tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighRes.csv')
print(tillage.columns)
print(tillage.head())
#tillage0=pd.read_csv(params.geopandasDataDir + 'Tillage0.csv')
#print(tillage0.head())
aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')
print(aez.columns)
print(aez.head())
print(aez.dtypes)

#print the value of each variable at the same index to make sure that coordinates align (they do)
print(pesticides.loc[6040])
print(fertilizer.loc[6040])
print(fertilizer_man.loc[6040])
#print(irrigation.loc[6040])
print(tillage.loc[6040])
print(aez.loc[6040])
print(wheat_yield.loc[6040])

#fertilizer is in kg/m² and fertilizer_man is in kg/km² while yield and pesticides are in kg/ha
#I would like to have all continuous variables in kg/ha
n_new = fertilizer['n'] * 10000
p_new = fertilizer['p'] * 10000
fert_new = pd.concat([n_new, p_new], axis='columns')
fert_new.rename(columns={'n':'n_kgha', 'p':'p_kgha'}, inplace=True)
fertilizer = pd.concat([fertilizer, fert_new], axis='columns') #kg/ha

applied_new = fertilizer_man['applied'] / 100
produced_new = fertilizer_man['produced'] / 100
man_new = pd.concat([applied_new, produced_new], axis='columns')
man_new.rename(columns={'applied':'applied_kgha', 'produced':'produced_kgha'}, inplace=True)
fertilizer_man = pd.concat([fertilizer_man, man_new], axis='columns') #kg/ha

#compile a combined factor for N including both N from fertilizer and manure
N_total = fertilizer['n_kgha'] + fertilizer_man['applied_kgha'] #kg/ha

'''
I don't remember what this was for
print(wheat_yield.columns.tolist())
l = wheat_yield.loc[:,'lats']
'''

#################################################################################
##############Loading variables without log to test the effect###################
#################################################################################
data_raw = {"lat": wheat_yield.loc[:,'lats'],
		"lon": wheat_yield.loc[:,'lons'],
		"area": wheat_yield.loc[:,'growArea'],
        "yield": wheat_yield.loc[:,'yield_kgPerHa'],
		"n_fertilizer": fertilizer.loc[:,'n_kgha'],
		"p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_total" : N_total,
        "pesticides_H": pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'maiz_is_mech'],
#        "irrigation": irrigation.loc[:,'area'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
dwheat_raw = pd.DataFrame(data=data_raw)
#select only the rows where the area of the cropland is larger than 0
dw0_raw=dwheat_raw.loc[dwheat_raw['area'] > 0]

test = dw0_raw.loc[dwheat_raw['thz_class'] == 0]
test1 = dw0_raw.loc[dwheat_raw['soil_class'] == 8]

#replace 0s in the moisture, climate and soil classes with NaN values so they
#can be handled with the .fillna method
dw0_raw['thz_class'] = dw0_raw['thz_class'].replace(0,np.nan)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(0,np.nan)
dw0_raw['soil_class'] = dw0_raw['soil_class'].replace(0,np.nan)
#NaN values throw errors in the regression, they need to be handled beforehand
#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
#this is fine for now as there most likely won't be any NaN values at full resolution
dw0_raw = dw0_raw.fillna(method='ffill')
#fill in the remaining couple of nans at the top of mechanized column
dw0_raw['mechanized'] = dw0_raw['mechanized'].fillna(1)

###############################################################################
############Loading log transformed values for all variables##################
##############################################################################


#using log values for the input into the regression
#unfortunately the ln of 0 is not defined
#just keeping the 0 would skew the results as that would imply a 1 in the data when there is a 0
#could just use the smallest value of the dataset as a substitute?
data_log = {"lat": wheat_yield.loc[:,'lats'],
		"lon": wheat_yield.loc[:,'lons'],
		"area": wheat_yield.loc[:,'growArea'],
        "yield": np.log(wheat_yield.loc[:,'yield_kgPerHa']),
		"n_fertilizer": np.log(fertilizer.loc[:,'n_kgha']),
		"p_fertilizer": np.log(fertilizer.loc[:,'p_kgha']),
        "n_manure": np.log(fertilizer_man.loc[:,'applied_kgha']),
        "n_total" : np.log(N_total),
        "pesticides_H": np.log(pesticides.loc[:,'total_H']),
        "mechanized": tillage.loc[:,'maiz_is_mech'],
#        "irrigation": np.log(irrigation.loc[:,'area']),
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}


dwheat_log = pd.DataFrame(data=data_log)
#select all rows from dwheat_log for which the column growArea has a value greater than zero
dw0_log=dwheat_log.loc[dwheat_log['area'] > 0]
#the data contains -inf values because the n+p+pests+irrigation columns contain 0 values for which ln(x) is not defined 
#calculate the minimum values for each column disregarding -inf values to see which is the lowest value in the dataset (excluding lat & lon)
min_dw0_log=dw0_log[dw0_log.iloc[:,3:11]>-inf].min()
#replace the -inf values with the minimum of the dataset + 5 : this is done to achieve a distinction between very small
#values and actual zeros
dw0_log.replace(-inf, -30, inplace=True)
#check distribution of AEZ factors in the historgrams
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

plt.hist(dw0_log['soil_class'], bins=50)
plt.hist(dw0_log['mst_class'], bins=50)
plt.hist(dw0_log['thz_class'], bins=50)
#ONLY RUN THIS BLOCK WHEN WORKING AT LOW RESOLUTION!
#AEZ factors contain unexpected 0s due to resolution rebinning
#urban class is missing in soil because of rebinning (urban class to small to dominant a large cell)
#convert 0s in the AEZ columns to NaN values so that they can be replaced by the ffill method
#0s make no sense in the dataset that is limited to wheat cropping area because the area is obviously on land
dw0_log['thz_class'] = dw0_log['thz_class'].replace(0,np.nan)
dw0_log['mst_class'] = dw0_log['mst_class'].replace(0,np.nan)
dw0_log['soil_class'] = dw0_log['soil_class'].replace(0,np.nan)
#NaN values throw errors in the regression, they need to be handled beforehand
#fill in the NaN vlaues in the dataset with a forward filling method (replacing NaN with the value in the cell before)
dw0_log = dw0_log.fillna(method='ffill')
#fill in the remaining couple of nans at the top of mechanized column
dw0_log['mechanized'] = dw0_log['mechanized'].fillna(1)

#Just some PLOTS

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot the continuous variables to get a sense of their distribution #RAW
plt.hist(dw0_raw['n_fertilizer'], bins=50)
plt.hist(dw0_raw['p_fertilizer'], bins=50)
plt.hist(dw0_raw['n_total'], bins=50)
plt.hist(dw0_raw['pesticides_H'], bins=100)
plt.hist(dw0_raw['irrigation'], bins=50)
'''
plt.ylim(0,5000)
plt.xlim(0, 0.04)
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
'''

#scatterplots for #RAW variables

dw0_raw.plot.scatter(x = 'n_fertilizer', y = 'yield')
dw0_raw.plot.scatter(x = 'p_fertilizer', y = 'yield')
dw0_raw.plot.scatter(x = 'pesticides_H', y = 'yield')
dw0_raw.plot.scatter(x = 'mechanized', y = 'yield')
dw0_raw.plot.scatter(x = 'non-mechanized', y = 'yield')
dw0_raw.plot.scatter(x = 'irrigation', y = 'yield')

#scatterplots and histograms for #LOG variables
dw0_log.plot.scatter(x = 'n_fertilizer', y = 'yield')
dw0_log.plot.scatter(x = 'p_fertilizer', y = 'yield')
dw0_log.plot.scatter(x = 'pesticides_H', y = 'yield')
dw0_log.plot.scatter(x = 'mechanized', y = 'yield')
dw0_log.plot.scatter(x = 'n_total', y = 'yield')
dw0_log.plot.scatter(x = 'irrigation', y = 'yield')
dw0_log.plot.scatter(x = 'thz_class', y = 'yield')
dw0_log.plot.scatter(x = 'mst_class', y = 'yield')
dw0_log.plot.scatter(x = 'soil_class', y = 'yield')

plt.hist(dw0_log['n_fertilizer'], bins=50)
plt.hist(dw0_log['p_fertilizer'], bins=50)
plt.hist(dw0_log['n_total'], bins=50)
plt.hist(dw0_log['pesticides_H'], bins=100)
plt.hist(dw0_log['irrigation'], bins=50)
plt.hist(dw0_log['mechanized'], bins=50)
plt.hist(dw0_log['thz_class'], bins=50)
plt.hist(dw0_log['mst_class'], bins=50)
plt.hist(dw0_log['soil_class'], bins=50)
plt.ylim(0,5000)
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
#####RAW##########
duw_mst_raw = pd.get_dummies(dw0_raw['mst_class'])
duw_thz_raw = pd.get_dummies(dw0_raw['thz_class'])
duw_soil_raw = pd.get_dummies(dw0_raw['soil_class'])
#####LOG##########
duw_mst_log = pd.get_dummies(dw0_log['mst_class'])
duw_thz_log = pd.get_dummies(dw0_log['thz_class'])
duw_soil_log = pd.get_dummies(dw0_log['soil_class'])
#rename the columns according to the classes
#####RAW##########
duw_mst_raw = duw_mst_raw.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
duw_thz_raw = duw_thz_raw.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
duw_soil_raw = duw_soil_raw.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#######LOG#########
duw_mst_log = duw_mst_log.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
duw_thz_log = duw_thz_log.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
duw_soil_log = duw_soil_log.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
####RAW#########
dwheat_d_raw = pd.concat([dw0_raw, duw_mst_raw, duw_thz_raw, duw_soil_raw], axis='columns')
######LOG#########
dwheat_d = pd.concat([dw0_log, duw_mst_log, duw_thz_log, duw_soil_log], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
#####RAW#####
dwheat_duw_raw = dwheat_d_raw.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')
########LOG#######
dwheat_duw_log = dwheat_d.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dwheat_val_raw = dwheat_duw_raw.sample(frac=0.2, random_state=2705) #RAW
dwheat_val_log = dwheat_duw_log.sample(frac=0.2, random_state=2705) #LOG
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dwheat_fit_raw = dwheat_duw_raw.drop(dwheat_val_raw.index) #RAW
dwheat_fit_log = dwheat_duw_log.drop(dwheat_val_log.index) #LOG

##################Collinearity################################

###########RAW#################

grid = sb.PairGrid(data= dwheat_fit_raw,
                    vars = ['n_fertilizer', 'p_fertilizer', 'n_total',
                    'pesticides_H', 'mechanized', 'irrigation'], height = 4)
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
                     edgecolor = 'k')
grid = grid.map_lower(sb.kdeplot, cmap = 'Reds')
#wanted to display the correlation coefficient in the lower triangle but don't know how
#grid = grid.map_lower(corr)

sb.pairplot(dwheat_duw_raw)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dwheat_cor_raw = dwheat_fit_raw.drop(['lat', 'lon', 'area', 'yield'], axis='columns')
#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(dwheat_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp = dwheat_cor_raw.corr(method='spearman')
print(sp)
sp.iloc[0,1:5]
sp.iloc[1,2:5]
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

X = add_constant(dwheat_cor_raw)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
#drop separate n variables
cor_n_total_raw = dwheat_cor_raw.drop(['n_fertilizer', 'n_manure', 'p_fertilizer', 'S4_moderate_lim', 'Trop_low'], axis='columns')
X1 = add_constant(cor_n_total_raw)
pd.Series([variance_inflation_factor(X1.values, i) 
               for i in range(X1.shape[1])], 
              index=X1.columns)
#if I leave p_fertilizer in, the respective VIF for p and n are a little over 5 but still fine I guess
#if I drop p_fertilizer they are all good
#if I drop S4_moderate_lim the soil variables are fine and if I drop Trop_low the temp variables are fine
#all with a VIF around 1
#I don't know what that means though and how to handle this with categorcial data

###########LOG##################


######################Regression##############################

#R-style formula
#doesn't work for some reason... I always get parsing errors and I don't know why
mod = smf.ols(formula=' yield ~ n_total + pesticides_H + mechanized + irrigation', data=dwheat_fit_raw)

mod = smf.ols(formula='yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dwheat_fit_raw)

#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dwheat_fit_raw, return_type='dataframe')


#define x and y dataframes
#Y containing only yield
mop = dw0_raw.iloc[:,3]
m_endog_raw = dwheat_fit_raw.iloc[:,3] #RAW
m_endog_log = dwheat_fit_log.iloc[:,3] #LOG
#X containing all variables
m_exog = dw0_raw.iloc[:,4]
m_exog_alln_raw = dwheat_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #RAW
m_exog_alln_log = dwheat_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #LOG
#test with n total and p
m_exog_np_raw = dwheat_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns') #RAW
m_exog_np_log = dwheat_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns')  #LOG
#test with n total without p
m_exog_n_log = dwheat_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #RAW
m_exog_n_raw = dwheat_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #LOG
#I will move forward with n_total and without p probably as they seem to be highly
#correlated

####testing regression
#determining the models
###RAW###
mod = sm.OLS(mop, m_exog)
mod_alln_raw = sm.OLS(m_endog_raw, m_exog_alln_raw)
mod_np_raw = sm.OLS(m_endog_raw, m_exog_np_raw)
mod_n_raw = sm.OLS(m_endog_raw, m_exog_n_log)
###LOG
mod_alln_log = sm.OLS(m_endog_log, m_exog_alln_log)
mod_np_log = sm.OLS(m_endog_log, m_exog_np_log)
mod_n_log = sm.OLS(m_endog_log, m_exog_n_raw)
####LOG DEPENDENT####
mod_alln_mix = sm.OLS(m_endog_log, m_exog_alln_raw)
mod_np_mix = sm.OLS(m_endog_log, m_exog_np_raw)
mod_n_mix = sm.OLS(m_endog_log, m_exog_n_log)

#fitting the models
#####RAW#####
mod_x = mod.fit()
mod_res_alln_raw = mod_alln_raw.fit(method='qr')
mod_res_np_raw = mod_np_raw.fit()
mod_res_n_raw = mod_n_raw.fit()
####LOG####
mod_res_alln_log = mod_alln_log.fit(method='qr')
mod_res_np_log = mod_np_log.fit()
mod_res_n_log = mod_n_log.fit()
####LOG DEPENDENT####
mod_res_alln_mix = mod_alln_mix.fit()
mod_res_np_mix = mod_np_mix.fit(method='qr')
mod_res_n_mix = mod_n_mix.fit()

#printing the results
print(mod_x.summary())
print(mod_res_alln_raw.summary())
print(mod_res_np_raw.summary())
print(mod_res_n_raw.summary())


print(mod_res_n_log.summary())

print(mod_res_alln_mix.summary())
print(mod_res_np_mix.summary())
print(mod_res_n_mix.summary())


##########RESIDUALS#############



plt.scatter(mod_res_n_raw.resid_pearson)
sb.residplot(x=m_exog_n_log, y=m_endog_log)





'''
#calculations on percentage of mechanized and non-mechanized, don't work anymore with the new tillage codification
mech = dwheat_nozero['mechanized']
mech_total = mech.sum()
nonmech = dwheat_nozero['non-mechanized']
non_mech_total = nonmech.sum()
total = mech_total + non_mech_total
mech_per = mech_total / total * 100
non_mech_per = non_mech_total / total * 100


#einfach nur für die iloc sachen drin
#drop lat, lon and area from the dataframe to only include relevant variables
dwheat_rg = dwheat_fit.iloc[:,[3,4,5,7,8,9,10]]
dwheat_pl = dwheat_fit.iloc[:,[4,5,7,8,9,10]]
dwheat_yield = dwheat_fit.iloc[:,3]

mod1 =sm.GLM(dwheat_yield, dwheat_pl, family=sm.families.Gamma())
#for some reason it works with Gaussian and Tweedie but not with Gamma or Inverse Gaussian... I really don't know why
mod_results = mod1.fit()
mod_res_alln_log = mod2.fit(method='qr')
    
'''
 


#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dwheat_rg, return_type='dataframe')


