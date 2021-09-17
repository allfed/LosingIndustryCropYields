'''

File containing the code to explore data and perform a multiple regression
on yield for maize
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
from statsmodels.graphics.factorplots import interaction_plot
from sklearn import cluster

params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for maize

maize_yield=pd.read_pickle(params.geopandasDataDir + 'MAIZCropYield.pkl')

#display first 5 rows of maize yield dataset
maize_yield.head()

#select all rows from maize_yield for which the column growArea has a value greater than zero
maize_nozero=maize_yield.loc[maize_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
maize_kgha=maize_nozero['yield_kgPerHa']
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
mmean=maize_kgha.mean()
mmeadian=maize_kgha.median()
msd=maize_kgha.std()
mvar=maize_kgha.var()
#calculate the mean with total production and area to check if the computed means align
mmean_total = ( maize_nozero['totalYield'].sum()) / (maize_nozero['growArea'].sum())
#the means do not align, probably due to the rebinning process
#calculate weighted mean (by area) of the yield colum
mmean_weighted = round(np.average(maize_kgha, weights=maize_nozero['growArea']),2)
#now they align!

#check the datatype of yield_kgPerHa and logarithmize the values
#logging is done to check the histogram and regression fit of the transformed values
maize_kgha.dtype
maize_kgha_log=np.log(maize_kgha)

#plot maize yield distribution in a histogram
plt.hist(maize_kgha, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yield_kgPerHa
plt.hist(maize_kgha_log, bins=50)

#test if area without zeros aligns with FAOSTAT harvested area
maize_area_ha = sum(maize_nozero['growArea'])
print(maize_area_ha)
#164569574.0937798
#164586904	#FAOSTAT area from 2010 for maize

#subplot for all histograms
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[0].hist(maize_kgha, bins=50)
axs[1].hist(maize_kgha_log, bins=50)


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
pdf_list = []
dist_list = []
param_dict ={"Values":[]}
#set x to bins in the range of the raw data
x = np.linspace(0.01,
                25500, 100)

###################################################################################
#####Testing of multiple distributions visually and using logLik, AIC and BIC######
###################################################################################

#Lognormal distribution
dist_list.append('lognorm')
#fit distribution to rice yield data to get values for the parameters
param1 = stats.lognorm.fit(maize_kgha)
#store the parameters in the initialized dictionary
param_dict["Values"].append(param1)
print(param1)
#use the parameters to calculate values for the probability density function 
#(pdf) of the distribution
pdf_fitted = stats.lognorm.pdf(x, *param1)
#calculate the logarithmized pdf to calculate statistical values for the fit
pdf_fitted_log = stats.lognorm.logpdf(maize_kgha, *param1)
#store the log pdf in the pdf list
pdf_list.append(pdf_fitted_log)
#plot the histogram of the yield data and the curve of the lognorm pdf
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.legend()
plt.show()


#Exponential distribution
dist_list.append('exponential')
#get parameters and store them in the dictionary
param2 = stats.expon.fit(maize_kgha)
param_dict["Values"].append(param2)
print(param2)
#calculate pdf
pdf_fitted2 = stats.expon.pdf(x, *param2)
#calculate log pdf and store it in the list
pdf_fitted_log2 = stats.expon.logpdf(maize_kgha, *param2)
pdf_list.append(pdf_fitted_log2)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.legend()
plt.show()


#Weibull distribution
dist_list.append('weibull')
#get parameters and store them in the dictionary
param3 = stats.weibull_min.fit(maize_kgha)
#param_list.append(param3)
param_dict["Values"].append(param3)
print(param3)
#calculate pdf
pdf_fitted3 = stats.weibull_min.pdf(x, *param3)
#calculate log pdf and store it in the list
pdf_fitted_log3 = stats.weibull_min.logpdf(maize_kgha, *param3)
pdf_list.append(pdf_fitted_log3)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.ylim(top=0.00025)
plt.legend()
plt.show()

#Normal distribution
dist_list.append('normal')
#get parameters and store them in the dictionary
param4 = stats.norm.fit(maize_kgha)
#param_list.append(param4)
param_dict["Values"].append(param4)
print(param4)
#calculate pdf
pdf_fitted4 = stats.norm.pdf(x, *param4)
#calculate log pdf and store it in the list
pdf_fitted_log4 = stats.norm.logpdf(maize_kgha, *param4)
pdf_list.append(pdf_fitted_log4)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted4, lw=2, label="Fitted normal distribution")
plt.legend()
plt.show()

#Halfnorm distribution
dist_list.append('halfnormal')
#get parameters and store them in the dictionary
param5 = stats.halfnorm.fit(maize_kgha)
#param_list.append(param5)
param_dict["Values"].append(param5)
print(param5)
#calculate pdf
pdf_fitted5 = stats.halfnorm.pdf(x, *param5)
#calculate log pdf and store it in the list
pdf_fitted_log5 = stats.halfnorm.logpdf(maize_kgha, *param5)
pdf_list.append(pdf_fitted_log5)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted5, lw=2, label="Fitted halfnormal distribution")
plt.legend()
plt.show()

#Gamma distribution
dist_list.append('Gamma')
#get parameters and store them in the dictionary
param6 = stats.gamma.fit(maize_kgha)
#param_list.append(param6)
param_dict["Values"].append(param6)
print(param6)
#calculate pdf
pdf_fitted6 = stats.gamma.pdf(x, *param6)
#calculate log pdf and store it in the list
pdf_fitted_log6 = stats.gamma.logpdf(maize_kgha, *param6)
pdf_list.append(pdf_fitted_log6)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted6, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()

#Inverse Gamma distribution
dist_list.append('Inverse Gamma')
#get parameters and store them in the dictionary
param7 = stats.invgamma.fit(maize_kgha)
#param_list.append(param5)
param_dict["Values"].append(param7)
print(param7)
#calculate pdf
pdf_fitted7 = stats.invgamma.pdf(x, *param7)
#calculate log pdf and store it in the list
pdf_fitted_log7 = stats.invgamma.logpdf(maize_kgha, *param7)
pdf_list.append(pdf_fitted_log7)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted7, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.show()

x1 = np.linspace(4,
                11, 100)
#Normal distribution on log values
dist_list.append('normal on log')
#get parameters and store them in the dictionary
param8 = stats.norm.fit(maize_kgha_log)
#param_list.append(param4)
param_dict["Values"].append(param8)
print(param8)
#calculate pdf
pdf_fitted8 = stats.norm.pdf(x1, *param8)
#calculate log pdf and store it in the list
pdf_fitted_log8 = stats.norm.logpdf(maize_kgha_log, *param8)
pdf_list.append(pdf_fitted_log8)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha_log, bins=50, density=True)
plt.plot(x1, pdf_fitted8, lw=2, label="Fitted normal distribution on log")
plt.legend()
plt.title('log Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.show()

#one in all plot
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(x, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.plot(x, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.plot(x, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.plot(x, pdf_fitted4, lw=2, label="Fitted Normal distribution")
plt.plot(x, pdf_fitted5, lw=2, label="Fitted Halfnormal distribution")
plt.plot(x, pdf_fitted6, lw=2, label="Fitted Gamma distribution")
plt.plot(x, pdf_fitted7, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=20000)
plt.show()


#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_list, pdf_list, param_dict)
#    Distribution  loglikelihood           AIC           BIC
#7 normal on log   -4282.862107   8579.724214   8622.636689
#4     halfnormal  -31626.712066  63267.424132  63310.336607
#1    exponential  -31710.693326  63435.386653  63478.299127
#6  Inverse Gamma  -31974.796947  63963.593894  64006.506369
#3         normal  -32394.095591  64802.191183  64845.103658
#0        lognorm  -38323.325928  76660.651855  76703.564330
#2        weibull  -39361.287338  78736.574675  78779.487150
#5          Gamma  -1.949240e+06  3.898495e+06  3.898544e+06
#best fit so far: normal on log values by far

'''
Load factor data and extract zeros
'''
pesticides=pd.read_pickle(params.geopandasDataDir + 'CornPesticidesByCrop.pkl')
print(pesticides.columns)
print(pesticides.head())
fertilizer=pd.read_pickle(params.geopandasDataDir + 'Fertilizer.pkl') #kg/m²
print(fertilizer.columns)
print(fertilizer.head())
fertilizer_man=pd.read_pickle(params.geopandasDataDir + 'FertilizerManure.pkl') #kg/km²
print(fertilizer_man.columns)
print(fertilizer_man.head())
irrigation=pd.read_pickle(params.geopandasDataDir + 'Irrigation.pkl')
print(irrigation.columns)
print(irrigation.head())
tillage=pd.read_pickle(params.geopandasDataDir + 'Tillage.pkl')
print(tillage.columns)
print(tillage.head())
#tillage0=pd.read_pickle(params.geopandasDataDir + 'Tillage0.pkl')
#print(tillage0.head())
aez=pd.read_pickle(params.geopandasDataDir + 'AEZ.pkl')
print(aez.columns)
print(aez.head())
print(aez.dtypes)

#print the value of each variable at the same index to make sure that coordinates align (they do)
print(pesticides.loc[6040])
print(fertilizer.loc[16592])
print(fertilizer_man.loc[16592])
print(irrigation.loc[6040])
print(tillage.loc[5682])
print(aez.loc[6041])
print(maize_yield.loc[6040])

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
print(maize_yield.columns.tolist())
l = maize_yield.loc[:,'lats']
'''

#################################################################################
##############Loading variables without log to test the effect###################
#################################################################################
data_raw = {"lat": maize_yield.loc[:,'lats'],
		"lon": maize_yield.loc[:,'lons'],
		"area": maize_yield.loc[:,'growArea'],
        "yield": maize_yield.loc[:,'yield_kgPerHa'],
		"n_fertilizer": fertilizer.loc[:,'n_kgha'],
		"p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_total" : N_total,
        "pesticides_H": pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'maiz_is_mech'],
        "irrigation": irrigation.loc[:,'area'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
dmaize_raw = pd.DataFrame(data=data_raw)
#select only the rows where the area of the cropland is larger than 0
dm0_raw=dmaize_raw.loc[dmaize_raw['area'] > 0]

#replace 0s in the moisture, climate and soil classes with NaN values so they
#can be handled with the .fillna method
dm0_raw['thz_class'] = dm0_raw['thz_class'].replace(0,np.nan)
dm0_raw['mst_class'] = dm0_raw['mst_class'].replace(0,np.nan)
dm0_raw['soil_class'] = dm0_raw['soil_class'].replace(0,np.nan)
#NaN values throw errors in the regression, they need to be handled beforehand
#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
#this is fine for now as there most likely won't be any NaN values at full resolution
dm0_raw = dm0_raw.fillna(method='ffill')
#fill in the remaining couple of nans at the top of mechanized column
dm0_raw['mechanized'] = dm0_raw['mechanized'].fillna(1)

###############################################################################
############Loading log transformed values for all variables##################
##############################################################################


#using log values for the input into the regression
#unfortunately the ln of 0 is not defined
#just keeping the 0 would skew the results as that would imply a 1 in the data when there is a 0
#could just use the smallest value of the dataset as a substitute?
data_log = {"lat": maize_yield.loc[:,'lats'],
		"lon": maize_yield.loc[:,'lons'],
		"area": maize_yield.loc[:,'growArea'],
        "yield": np.log(maize_yield.loc[:,'yield_kgPerHa']),
		"n_fertilizer": np.log(fertilizer.loc[:,'n_kgha']),
		"p_fertilizer": np.log(fertilizer.loc[:,'p_kgha']),
        "n_manure": np.log(fertilizer_man.loc[:,'applied_kgha']),
        "n_total" : np.log(N_total),
        "pesticides_H": np.log(pesticides.loc[:,'total_H']),
        "mechanized": tillage.loc[:,'maiz_is_mech'],
        "irrigation": np.log(irrigation.loc[:,'area']),
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}


dmaize_log = pd.DataFrame(data=data_log)
#select all rows from dmaize_log for which the column growArea has a value greater than zero
dm0_log=dmaize_log.loc[dmaize_log['area'] > 0]
#the data contains -inf values because the n+p+pests+irrigation columns contain 0 values for which ln(x) is not defined 
#calculate the minimum values for each column disregarding -inf values to see which is the lowest value in the dataset (excluding lat & lon)
min_dm0_log=dm0_log[dm0_log.iloc[:,3:11]>-inf].min()
#replace the -inf values with the minimum of the dataset + 5 : this is done to achieve a distinction between very small
#values and actual zeros
dm0_log.replace(-inf, -30, inplace=True)
#check distribution of AEZ factors in the historgrams
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

plt.hist(dm0_log['soil_class'], bins=50)
plt.hist(dm0_log['mst_class'], bins=50)
plt.hist(dm0_log['thz_class'], bins=50)
#ONLY RUN THIS BLOCK WHEN WORKING AT LOW RESOLUTION!
#AEZ factors contain unexpected 0s due to resolution rebinning
#urban class is missing in soil because of rebinning (urban class to small to dominant a large cell)
#convert 0s in the AEZ columns to NaN values so that they can be replaced by the ffill method
#0s make no sense in the dataset that is limited to maize cropping area because the area is obviously on land
dm0_log['thz_class'] = dm0_log['thz_class'].replace(0,np.nan)
dm0_log['mst_class'] = dm0_log['mst_class'].replace(0,np.nan)
dm0_log['soil_class'] = dm0_log['soil_class'].replace(0,np.nan)
#NaN values throw errors in the regression, they need to be handled beforehand
#fill in the NaN vlaues in the dataset with a forward filling method (replacing NaN with the value in the cell before)
dm0_log = dm0_log.fillna(method='ffill')
#fill in the remaining couple of nans at the top of mechanized column
dm0_log['mechanized'] = dm0_log['mechanized'].fillna(1)

#Just some PLOTS

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot the continuous variables to get a sense of their distribution #RAW
plt.hist(dm0_raw['n_fertilizer'], bins=50)
plt.hist(dm0_raw['p_fertilizer'], bins=50)
plt.hist(dm0_raw['n_total'], bins=50)
plt.hist(dm0_raw['pesticides_H'], bins=100)
plt.hist(dm0_raw['irrigation'], bins=50)
'''
plt.ylim(0,5000)
plt.xlim(0, 0.04)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
'''

#scatterplots for #RAW variables

dm0_raw.plot.scatter(x = 'n_fertilizer', y = 'yield')
dm0_raw.plot.scatter(x = 'p_fertilizer', y = 'yield')
dm0_raw.plot.scatter(x = 'pesticides_H', y = 'yield')
dm0_raw.plot.scatter(x = 'mechanized', y = 'yield')
dm0_raw.plot.scatter(x = 'non-mechanized', y = 'yield')
dm0_raw.plot.scatter(x = 'irrigation', y = 'yield')

#scatterplots and histograms for #LOG variables
dm0_log.plot.scatter(x = 'n_fertilizer', y = 'yield')
dm0_log.plot.scatter(x = 'p_fertilizer', y = 'yield')
dm0_log.plot.scatter(x = 'pesticides_H', y = 'yield')
dm0_log.plot.scatter(x = 'mechanized', y = 'yield')
dm0_log.plot.scatter(x = 'n_total', y = 'yield')
dm0_log.plot.scatter(x = 'irrigation', y = 'yield')
dm0_log.plot.scatter(x = 'thz_class', y = 'yield')
dm0_log.plot.scatter(x = 'mst_class', y = 'yield')
dm0_log.plot.scatter(x = 'soil_class', y = 'yield')

plt.hist(dm0_log['n_fertilizer'], bins=50)
plt.hist(dm0_log['p_fertilizer'], bins=50)
plt.hist(dm0_log['n_total'], bins=50)
plt.hist(dm0_log['pesticides_H'], bins=100)
plt.hist(dm0_log['irrigation'], bins=50)
plt.hist(dm0_log['mechanized'], bins=50)
plt.hist(dm0_log['thz_class'], bins=50)
plt.hist(dm0_log['mst_class'], bins=50)
plt.hist(dm0_log['soil_class'], bins=50)
plt.ylim(0,5000)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
#####RAW##########
dum_mst_raw = pd.get_dummies(dm0_raw['mst_class'])
dum_thz_raw = pd.get_dummies(dm0_raw['thz_class'])
dum_soil_raw = pd.get_dummies(dm0_raw['soil_class'])
#####LOG##########
dum_mst_log = pd.get_dummies(dm0_log['mst_class'])
dum_thz_log = pd.get_dummies(dm0_log['thz_class'])
dum_soil_log = pd.get_dummies(dm0_log['soil_class'])
#rename the columns according to the classes
#####RAW##########
dum_mst_raw = dum_mst_raw.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
dum_thz_raw = dum_thz_raw.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
dum_soil_raw = dum_soil_raw.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#######LOG#########
dum_mst_log = dum_mst_log.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
dum_thz_log = dum_thz_log.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
dum_soil_log = dum_soil_log.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
####RAW#########
dmaize_d_raw = pd.concat([dm0_raw, dum_mst_raw, dum_thz_raw, dum_soil_raw], axis='columns')
######LOG#########
dmaize_d = pd.concat([dm0_log, dum_mst_log, dum_thz_log, dum_soil_log], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dmaize_val_raw = dmaize_d_raw.sample(frac=0.2, random_state=2705) #RAW
dmaize_val_log = dmaize_d.sample(frac=0.2, random_state=2705) #LOG
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dmaize_fit_raw = dmaize_d_raw.drop(dmaize_val_raw.index) #RAW
dmaize_fit_log = dmaize_d.drop(dmaize_val_log.index) #LOG

#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
#####RAW#####
dmaize_fitd_raw = dmaize_fit_raw.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')
########LOG#######
dmaize_fitd_log = dmaize_fit_log.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')

##################Collinearity################################

###########RAW#################

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dmaize_cor_raw = dmaize_fit_raw.drop(['lat', 'lon', 'area', 'yield'], axis='columns')

#wanted to display the correlation coefficient in the lower triangle but don't know how
#so I calculated the correlation heatmap separately below
grid = sb.PairGrid(data= dmaize_cor_raw,
                    vars = ['n_fertilizer', 'p_fertilizer', 'n_total',
                    'pesticides_H', 'irrigation', 'mechanized','thz_class',
                    'mst_class', 'soil_class'], 
                    height = 4)
grid = grid.map_upper(sb.scatterplot, color = 'darkred')
grid = grid.map_diag(sb.histplot, bins = 10, color = 'darkred', 
                     edgecolor = 'k')
grid = grid.map_lower(sb.kdeplot)

#this can also be achieved easier with pairplot
sb.pairplot(dmaize_fitd_raw)

#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(dmaize_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp = dmaize_cor_raw.corr(method='spearman')

#define a mask to only get the lower triangle of the correlation matrix
mask = np.triu(np.ones_like(dmaize_cor_raw.corr(), dtype=np.bool))
#calculate a heatmap according to correlation between the variables
mheat = sb.heatmap(dmaize_cor_raw.corr(method='spearman'), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
mheat.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
mheat
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

agglo = cluster.FeatureAgglomeration(n_clusters=20, compute_distances=True)
clus = agglo.fit(dmaize_cor_raw)
scipy.cluster.hierarchy.dendrogram(clus)

############Variance inflation factor##########################

X = add_constant(dmaize_cor_raw)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
#drop separate n variables
cor_n_total_raw = dmaize_cor_raw.drop(['n_fertilizer', 'n_manure', 'p_fertilizer',
                                       #'S4_moderate_lim', 'Trop_low'
                                       ], axis='columns')
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

##############Interactions##########################
n=dmaize_d_raw['n_total'].reset_index(drop=True)
m=dmaize_d_raw['mechanized'].reset_index(drop=True)
i=dmaize_d_raw['irrigation'].reset_index(drop=True)
s=dmaize_d_raw['soil_class'].reset_index(drop=True)
mst=dmaize_d_raw['mst_class'].reset_index(drop=True)
t=dmaize_d_raw['thz_class'].reset_index(drop=True)
p=dmaize_fit_raw['pesticides_H'].reset_index(drop=True)
y=dmaize_d_raw['yield'].reset_index(drop=True)

data_i = {"n": n,
		"m": m,
		"i": i,
        "s": s,
		"mst": mst,
		"t": t,
        "p": p,
        "y" :y,
		}
intpl = pd.DataFrame(data=data_i)

fig = interaction_plot(n, m, y, plottype='scatter')
sb.lmplot(y='y', x='mst', hue='m', data=intpl)
plt.xlim(0,8)
plt.show()


######################Regression##############################

#R-style formula
#doesn't work for some reason... I always get parsing errors and I don't know why
mod = smf.ols(formula='yield ~ n_total * pesticides_H', data=dmaize_fit_raw)
mod1 = smf.ols(formula='n_total ~ pesticides_H + mechanized + irrigation', data=dmaize_fit_raw)

dmaize_fit_raw = dmaize_fit_raw.rename(columns={'yield':'Y'}, errors='raise')
mod = smf.ols(formula='Y ~ n_total + pesticides_H + irrigation + mechanized', data=dmaize_fit_raw)
m = mod.fit()
print(m.summary())

mod24 = smf.glm(formula='Y ~ 1', data=dmaize_fit_raw, family=sm.families.Gamma())
m24 = mod24.fit()
print(m24.summary())

dm0_raw2 = dm0_raw.rename(columns={'yield':'Y'}, errors='raise')
dm0_log2 = dm0_log.rename(columns={'yield':'Y'}, errors='raise')
modas = smf.glm(formula='Y ~ n_total + pesticides_H + irrigation + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dm0_raw2, 
              family=sm.families.Gamma())

pseudoR = 1-(1575.3/1722.4)    

mas = modas.fit()
print(mas.summary())
modlog = smf.ols(formula='Y ~ n_total * pesticides_H * irrigation * mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dm0_log2)
mlog = modlog.fit()
print(mlog.summary())
modraw = smf.ols(formula='Y ~ n_total * pesticides_H * irrigation * mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dm0_raw2)
mraw = modraw.fit()
print(mraw.summary())

print(type(dmaize_fit_raw.loc[6040,'yield']))
print(type(dmaize_fit_raw.loc[6040,'n_total']))
print(type(dmaize_fit_raw.loc[6040,'pesticides_H']))
print(type(dmaize_fit_raw.loc[6040,'mechanized']))
print(type(dmaize_fit_raw.loc[6040,'irrigation']))
pd.DataFrame(dmaize_fit_raw['yield']).applymap(type).apply(pd.value_counts).fillna(0)

mod = smf.ols(formula='yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dmaize_fit_raw)

#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dmaize_fit_raw, return_type='dataframe')


#define x and y dataframes
#Y containing only yield
m_endog_raw = dmaize_fit_raw.iloc[:,3] #RAW
m_endog_log = dmaize_fit_log.iloc[:,3] #LOG
#X containing all variables
m_exog_alln_raw = dmaize_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #RAW
m_exog_alln_log = dmaize_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #LOG
#test with n total and p
m_exog_np_raw = dmaize_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns') #RAW
m_exog_np_log = dmaize_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns')  #LOG
#test with n total without p
m_exog_n_log = dmaize_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #RAW
m_exog_n_raw = dmaize_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #LOG
#I will move forward with n_total and without p probably as they seem to be highly
#correlated

####testing regression
#determining the models
###RAW###
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
print(mod_res_alln_raw.summary())
print(mod_res_np_raw.summary())
print(mod_res_n_raw.summary())


print(mod_res_n_log.summary())

print(mod_res_alln_mix.summary())
print(mod_res_np_mix.summary())
print(mod_res_n_mix.summary())

#####################Outliers####################################

############RAW#################################
dmaize_d_raw.plot.scatter(x = 'n_fertilizer', y = 'yield')
plt.hist(dmaize_d_raw['yield'])
print(dmaize_d_raw['yield'].max())
print(dmaize_d_raw['yield'].quantile(0.999))
dmaize_q_raw=dmaize_d_raw.loc[dmaize_d_raw['yield'] < dmaize_d_raw['yield'].quantile(0.99)]
dmaize_q_raw.plot.scatter(x = 'n_fertilizer', y = 'yield')
print(dmaize_d_raw['n_fertilizer'].max())
print(dmaize_d_raw['n_fertilizer'].quantile(0.999))
dmaize_q_raw=dmaize_q_raw.loc[dmaize_q_raw['n_fertilizer'] < dmaize_q_raw['n_fertilizer'].quantile(0.99)]
dmaize_q_raw.plot.scatter(x = 'n_fertilizer', y = 'yield')
plt.hist(dmaize_q_raw['yield'], bins=50)

################LOG############################
dmaize_d.plot.scatter(x = 'n_fertilizer', y = 'yield')
plt.hist(dmaize_d['yield'])
print(dmaize_d['yield'].max())
print(dmaize_d['yield'].quantile(0.99))
dmaize_q_log=dmaize_d.loc[dmaize_d['yield'] < dmaize_d['yield'].quantile(0.99)]
dmaize_q_log.plot.scatter(x = 'n_fertilizer', y = 'yield')
print(dmaize_d['n_fertilizer'].min())
print(dmaize_d['n_fertilizer'].quantile(0.095))
dmaize_q_log=dmaize_q_log.loc[dmaize_q_log['n_fertilizer'] > dmaize_q_log['n_fertilizer'].quantile(0.095)]
dmaize_q_log.plot.scatter(x = 'n_fertilizer', y = 'yield')
plt.hist(dmaize_q_log['yield'], bins=50)


##########RESIDUALS#############



plt.scatter(mod_res_n_raw.resid_pearson)
sb.residplot(x=m_exog_n_log, y=m_endog_log)





'''
#calculations on percentage of mechanized and non-mechanized, don't work anymore with the new tillage codification
mech = dmaize_nozero['mechanized']
mech_total = mech.sum()
nonmech = dmaize_nozero['non-mechanized']
non_mech_total = nonmech.sum()
total = mech_total + non_mech_total
mech_per = mech_total / total * 100
non_mech_per = non_mech_total / total * 100


#einfach nur für die iloc sachen drin
#drop lat, lon and area from the dataframe to only include relevant variables
dmaize_rg = dmaize_fit.iloc[:,[3,4,5,7,8,9,10]]
dmaize_pl = dmaize_fit.iloc[:,[4,5,7,8,9,10]]
dmaize_yield = dmaize_fit.iloc[:,3]

mod1 =sm.GLM(dmaize_yield, dmaize_pl, family=sm.families.Gamma())
#for some reason it works with Gaussian and Tweedie but not with Gamma or Inverse Gaussian... I really don't know why
mod_results = mod1.fit()
mod_res_alln_log = mod2.fit(method='qr')
    
'''
 


#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dmaize_rg, return_type='dataframe')


