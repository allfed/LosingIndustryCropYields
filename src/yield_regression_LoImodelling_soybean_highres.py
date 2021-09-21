'''

File containing the code to explore data and perform a multiple regression
on yield for soyb
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

#import yield geopandas data for soyb

soyb_yield=pd.read_csv(params.geopandasDataDir + 'SOYBCropYieldHighRes.csv')

#display first 5 rows of soyb yield dataset
soyb_yield.head()

#select all rows from soyb_yield for which the column growArea has a value greater than zero
soyb_nozero=soyb_yield.loc[soyb_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
soyb_kgha=soyb_nozero['yield_kgPerHa']
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
smean=soyb_kgha.mean()
smeadian=soyb_kgha.median()
ssd=soyb_kgha.std()
svar=soyb_kgha.var()
smax=soyb_kgha.max()
#calculate the mean with total production and area to check if the computed means align
smean_total = ( soyb_nozero['totalYield'].sum()) / (soyb_nozero['growArea'].sum())
#the means do not align, probably due to the rebinning process
#calculate weighted mean (by area) of the yield colum
smean_weighted = round(np.average(soyb_kgha, weights=soyb_nozero['growArea']),2)
#now they align!

#check the datatype of yield_kgPerHa and logarithmize the values
#logging is done to check the histogram and regoression fit of the transformed values
soyb_kgha.dtype
soyb_kgha_log=np.log(soyb_kgha)

#plot soyb yield distribution in a histogram
plt.hist(soyb_kgha, bins=50)
plt.title('soyb yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yield_kgPerHa
plt.hist(soyb_kgha_log, bins=50)

#test if area without zeros aligns with FAOSTAT harvested area
soyb_area_ha = sum(soyb_nozero['growArea'])
print(soyb_area_ha)
#164569574.0937798
#164586904	#FAOSTAT area from 2010 for soyb

#subplot for all histograms
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[0].hist(soyb_kgha, bins=50)
axs[1].hist(soyb_kgha_log, bins=50)


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
pdf_lists = []
dist_lists = []
param_dicts ={"Values":[]}
#set xs to bins in the range of the raw data
xs = np.linspace(0.01,
                12500, 100)

###################################################################################
#####Testing of multiple distributions visually and using logLik, AIC and BIC######
###################################################################################

#Exponential distribution
dist_lists.append('exponential')
#get parameters and store them in the dictionary
param1 = stats.expon.fit(soyb_kgha)
param_dicts["Values"].append(param1)
print(param1)
#calculate pdf
pdf_fitted1 = stats.expon.pdf(xs, *param1)
#calculate log pdf and store it in the list
pdf_fitted_log1 = stats.expon.logpdf(soyb_kgha, *param1)
pdf_lists.append(pdf_fitted_log1)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(xs, pdf_fitted1, lw=2, label="Fitted Exponential distribution")
plt.legend()
plt.show()

#Normal distribution
dist_lists.append('normal')
#get parameters and store them in the dictionary
param2 = stats.norm.fit(soyb_kgha)
#param_list.append(param2)
param_dicts["Values"].append(param2)
print(param2)
#calculate pdf
pdf_fitted2 = stats.norm.pdf(xs, *param2)
#calculate log pdf and store it in the list
pdf_fitted_log2 = stats.norm.logpdf(soyb_kgha, *param2)
pdf_lists.append(pdf_fitted_log2)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(xs, pdf_fitted2, lw=2, label="Fitted normal distribution")
plt.legend()
plt.show()

#Gamma distribution
dist_lists.append('Gamma')
#get parameters and store them in the dictionary
param3 = stats.gamma.fit(soyb_kgha)
#param_list.append(param3)
param_dicts["Values"].append(param3)
print(param3)
#calculate pdf
pdf_fitted3 = stats.gamma.pdf(xs, *param3)
#calculate log pdf and store it in the list
pdf_fitted_log3 = stats.gamma.logpdf(soyb_kgha, *param3)
pdf_lists.append(pdf_fitted_log3)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(xs, pdf_fitted3, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()

#Inverse Gamma distribution
dist_lists.append('Inverse Gamma')
#get parameters and store them in the dictionary
param4 = stats.invgamma.fit(soyb_kgha)
#param_list.append(param4)
param_dicts["Values"].append(param4)
print(param4)
#calculate pdf
pdf_fitted4 = stats.invgamma.pdf(xs, *param4)
#calculate log pdf and store it in the list
pdf_fitted_log4 = stats.invgamma.logpdf(soyb_kgha, *param4)
pdf_lists.append(pdf_fitted_log4)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(xs, pdf_fitted4, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.show()

xs1 = np.linspace(4,
                11, 100)
#Normal distribution on log values
dist_lists.append('normal on log')
#get parameters and store them in the dictionary
param5 = stats.norm.fit(soyb_kgha_log)
#param_list.append(param5)
param_dicts["Values"].append(param5)
print(param5)
#calculate pdf
pdf_fitted5 = stats.norm.pdf(xs1, *param5)
#calculate log pdf and store it in the list
pdf_fitted_log5 = stats.norm.logpdf(soyb_kgha_log, *param5)
pdf_lists.append(pdf_fitted_log5)
#plot data histogram and pdf curve
h = plt.hist(soyb_kgha_log, bins=50, density=True)
plt.plot(xs1, pdf_fitted5, lw=2, label="Fitted normal distribution on log")
plt.legend()
plt.title('log soyb yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.show()

#one in all plot
h = plt.hist(soyb_kgha, bins=50, density=True)
plt.plot(xs, pdf_fitted1, lw=2, label="Fitted Exponential distribution")
plt.plot(xs, pdf_fitted2, lw=2, label="Fitted Normal distribution")
plt.plot(xs, pdf_fitted3, lw=2, label="Fitted Gamma distribution")
plt.plot(xs, pdf_fitted4, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.title('soyb yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=20000)
plt.show()


#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_lists, pdf_lists, param_dicts)
'''
    Distribution  loglikelihood           AIC           BIC
7  normal on log  -3.748763e+05  7.497686e+05  7.498546e+05
6  Inverse Gamma  -2.859205e+06  5.718427e+06  5.718513e+06
5          Gamma  -2.860227e+06  5.720469e+06  5.720555e+06
3         normal  -2.908569e+06  5.817154e+06  5.817240e+06
1    exponential  -2.910045e+06  5.820106e+06  5.820192e+06
0        lognorm  -3.587555e+06  7.175125e+06  7.175211e+06
2        weibull  -3.694327e+06  7.388671e+06  7.388757e+06
4     halfnormal           -inf           inf           inf
Best fit is normal on log by far, then inverse gamma on non-log
'''

'''
Load factor data and extract zeros
'''
s_pesticides=pd.read_csv(params.geopandasDataDir + 'SoybeanPesticidesHighRes.csv')
print(s_pesticides.columns)
print(s_pesticides.head())
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
s_tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighRessoyb.csv')
print(s_tillage.columns)
print(s_tillage.head())
#s_tillage0=pd.read_csv(params.geopandasDataDir + 'Tillage0.csv')
#print(s_tillage0.head())
aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')
print(aez.columns)
print(aez.head())
print(aez.dtypes)

#print the value of each variable at the same index to make sure that coordinates align (they do)
print(s_pesticides.loc[6040])
print(fertilizer.loc[6040])
print(fertilizer_man.loc[6040])
#print(irrigation.loc[6040])
print(s_tillage.loc[6040])
print(aez.loc[6040])
print(soyb_yield.loc[6040])

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
print(soyb_yield.columns.tolist())
l = soyb_yield.loc[:,'lats']
'''

#################################################################################
##############Loading variables without log to test the effect###################
#################################################################################
data_raw = {"lat": soyb_yield.loc[:,'lats'],
		"lon": soyb_yield.loc[:,'lons'],
		"area": soyb_yield.loc[:,'growArea'],
        "yield": soyb_yield.loc[:,'yield_kgPerHa'],
		"n_fertilizer": fertilizer.loc[:,'n_kgha'],
		"p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_total" : N_total,
        "pesticides_H": s_pesticides.loc[:,'total_H'],
        "mechanized": s_tillage.loc[:,'soyb_is_mech'],
#        "irrigation": irrigation.loc[:,'area'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
dsoyb_raw = pd.DataFrame(data=data_raw)
#select only the rows where the area of the cropland is larger than 0
ds0_raw=dsoyb_raw.loc[dsoyb_raw['area'] > 0]

#test if there are cells with 0s for the AEZ classes (there shouldn't be any)
s_testt = ds0_raw.loc[ds0_raw['thz_class'] == 0] #only 25 0s
s_testm = ds0_raw.loc[ds0_raw['mst_class'] == 0] #only 25 0s
s_tests = ds0_raw.loc[ds0_raw['soil_class'] == 0]
#1279 0s probably due to the original soil dataset being in 30 arcsec resolution:
    #land/ocean boundaries, especially of islands, don't always align perfectly

#test if certain classes of the AEZ aren't present in the dataset because they
#represent conditions which aren't beneficial for plant growth
#thz_class: test Arctic and Bor_cold_with_permafrost
s_test_t9 = ds0_raw.loc[ds0_raw['thz_class'] == 9]
#31 with Boreal and permafrost: reasonable
s_test_t10 = ds0_raw.loc[ds0_raw['thz_class'] == 10]
#60 with Arctic: is reasonable

#mst_class: test LPG<60days
s_test_m = ds0_raw.loc[ds0_raw['mst_class'] == 1]
#2676 in LPG<60 days class: probably due to irrigation

#soil class: test urban, water bodies and very steep class
s_test_s1 = ds0_raw.loc[ds0_raw['soil_class'] == 1]
#7852 in very steep class: makes sense, there is marginal agriculture in
#agricultural outskirts
s_test_s7 = ds0_raw.loc[ds0_raw['soil_class'] == 7]
#2280 in water class: this doesn't make sense but also due to resolution
#I think these should be substituted
s_test_s8 = ds0_raw.loc[ds0_raw['soil_class'] == 8]
#2372 in urban class: probably due to finer resolution in soil class, e.g. course of 
#the Nile is completely classified with yield estimates even though there are many urban areas
#Question: should the urban datapoints be taken out due to them being unreasonable? But then again
#the other datasets most likely contain values in these spots as well (equally unprecise), so I would
#just lose information
#I could substitute them like the water bodies

#test mech dataset values
s_test_mech0 = ds0_raw.loc[ds0_raw['mechanized'] == 0] #82541
s_test_mech1 = ds0_raw.loc[ds0_raw['mechanized'] == 1] #172097
s_test_mechn = ds0_raw.loc[ds0_raw['mechanized'] == -9] #90658
#this is a problem: -9 is used as NaN value and there are way, way too many

s_test_f = ds0_raw.loc[ds0_raw['n_fertilizer'] == 0] #15279
s_test_pf = ds0_raw.loc[ds0_raw['p_fertilizer'] == 0] #15975
s_test_man = ds0_raw.loc[ds0_raw['n_manure'] == 0] #9794
s_test_p = ds0_raw.loc[ds0_raw['pesticides_H'] == 0] #183822 335589

#replace 0s in the moisture, climate and soil classes with NaN values so they
#can be handled with the .fillna method
ds0_raw['thz_class'] = ds0_raw['thz_class'].replace(0,np.nan)
ds0_raw['mst_class'] = ds0_raw['mst_class'].replace(0,np.nan)
ds0_raw['soil_class'] = ds0_raw['soil_class'].replace(0,np.nan)
#NaN values throw errors in the regression, they need to be handled beforehand
#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
#this is fine for now as there most likely won't be any NaN values at full resolution
ds0_raw = ds0_raw.fillna(method='ffill')
#fill in the remaining couple of nans at the top of mechanized column
ds0_raw['mechanized'] = ds0_raw['mechanized'].fillna(1)

###############################################################################
############Loading log transformed values for all variables##################
##############################################################################


#using log values for the input into the regression
#unfortunately the ln of 0 is not defined
#just keeping the 0 would skew the results as that would imply a 1 in the data when there is a 0
#could just use the smallest value of the dataset as a substitute?
data_log = {"lat": soyb_yield.loc[:,'lats'],
		"lon": soyb_yield.loc[:,'lons'],
		"area": soyb_yield.loc[:,'growArea'],
        "yield": np.log(soyb_yield.loc[:,'yield_kgPerHa']),
		"n_fertilizer": np.log(fertilizer.loc[:,'n_kgha']),
		"p_fertilizer": np.log(fertilizer.loc[:,'p_kgha']),
        "n_manure": np.log(fertilizer_man.loc[:,'applied_kgha']),
        "n_total" : np.log(N_total),
        "pesticides_H": np.log(s_pesticides.loc[:,'total_H']),
        "mechanized": s_tillage.loc[:,'soyb_is_mech'],
#        "irrigation": np.log(irrigation.loc[:,'area']),
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}


dsoyb_log = pd.DataFrame(data=data_log)
#select all rows from dsoyb_log for which the column growArea has a value greater than zero
ds0_log=dsoyb_log.loc[dsoyb_log['area'] > 0]
#the data contains -inf values because the n+p+pests+irrigation columns contain 0 values for which ln(x) is not defined 
#calculate the minimum values for each column disregarding -inf values to see which is the lowest value in the dataset (excluding lat & lon)
min_ds0_log=ds0_log[ds0_log.iloc[:,3:11]>-inf].min()
#replace the -inf values with the minimum of the dataset + 5 : this is done to achieve a distinction between very small
#values and actual zeros
ds0_log.replace(-inf, -30, inplace=True)
#check distribution of AEZ factors in the historgrams
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

plt.hist(ds0_log['soil_class'], bins=50)
plt.hist(ds0_log['mst_class'], bins=50)
plt.hist(ds0_log['thz_class'], bins=50)
#ONLY RUN THIS BLOCK WHEN WORKING AT LOW RESOLUTION!
#AEZ factors contain unexpected 0s due to resolution rebinning
#urban class is missing in soil because of rebinning (urban class to small to dominant a large cell)
#convert 0s in the AEZ columns to NaN values so that they can be replaced by the ffill method
#0s make no sense in the dataset that is limited to soyb cropping area because the area is obviously on land
ds0_log['thz_class'] = ds0_log['thz_class'].replace(0,np.nan)
ds0_log['mst_class'] = ds0_log['mst_class'].replace(0,np.nan)
ds0_log['soil_class'] = ds0_log['soil_class'].replace(0,np.nan)
#NaN values throw errors in the regression, they need to be handled beforehand
#fill in the NaN vlaues in the dataset with a forward filling method (replacing NaN with the value in the cell before)
ds0_log = ds0_log.fillna(method='ffill')
#fill in the remaining couple of nans at the top of mechanized column
ds0_log['mechanized'] = ds0_log['mechanized'].fillna(1)

#Just some PLOTS

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot the continuous variables to get a sense of their distribution #RAW
plt.hist(ds0_raw['n_fertilizer'], bins=50)
plt.hist(ds0_raw['p_fertilizer'], bins=50)
plt.hist(ds0_raw['n_total'], bins=50)
plt.hist(ds0_raw['pesticides_H'], bins=100)
plt.hist(ds0_raw['irrigation'], bins=50)
'''
plt.ylim(0,5000)
plt.xlim(0, 0.04)
plt.title('soyb yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
'''

#scatterplots for #RAW variables

ds0_raw.plot.scatter(x = 'n_fertilizer', y = 'yield')
ds0_raw.plot.scatter(x = 'p_fertilizer', y = 'yield')
ds0_raw.plot.scatter(x = 'pesticides_H', y = 'yield')
ds0_raw.plot.scatter(x = 'mechanized', y = 'yield')
ds0_raw.plot.scatter(x = 'non-mechanized', y = 'yield')
ds0_raw.plot.scatter(x = 'irrigation', y = 'yield')

#scatterplots and histograms for #LOG variables
ds0_log.plot.scatter(x = 'n_fertilizer', y = 'yield')
ds0_log.plot.scatter(x = 'p_fertilizer', y = 'yield')
ds0_log.plot.scatter(x = 'pesticides_H', y = 'yield')
ds0_log.plot.scatter(x = 'mechanized', y = 'yield')
ds0_log.plot.scatter(x = 'n_total', y = 'yield')
ds0_log.plot.scatter(x = 'irrigation', y = 'yield')
ds0_log.plot.scatter(x = 'thz_class', y = 'yield')
ds0_log.plot.scatter(x = 'mst_class', y = 'yield')
ds0_log.plot.scatter(x = 'soil_class', y = 'yield')

plt.hist(ds0_log['n_fertilizer'], bins=50)
plt.hist(ds0_log['p_fertilizer'], bins=50)
plt.hist(ds0_log['n_total'], bins=50)
plt.hist(ds0_log['pesticides_H'], bins=100)
plt.hist(ds0_log['irrigation'], bins=50)
plt.hist(ds0_log['mechanized'], bins=50)
plt.hist(ds0_log['thz_class'], bins=50)
plt.hist(ds0_log['mst_class'], bins=50)
plt.hist(ds0_log['soil_class'], bins=50)
plt.ylim(0,5000)
plt.title('soyb yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
#####RAW##########
dus_mst_raw = pd.get_dummies(ds0_raw['mst_class'])
dus_thz_raw = pd.get_dummies(ds0_raw['thz_class'])
dus_soil_raw = pd.get_dummies(ds0_raw['soil_class'])
#####LOG##########
dus_mst_log = pd.get_dummies(ds0_log['mst_class'])
dus_thz_log = pd.get_dummies(ds0_log['thz_class'])
dus_soil_log = pd.get_dummies(ds0_log['soil_class'])
#rename the columns according to the classes
#####RAW##########
dus_mst_raw = dus_mst_raw.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
dus_thz_raw = dus_thz_raw.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
dus_soil_raw = dus_soil_raw.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#######LOG#########
dus_mst_log = dus_mst_log.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
dus_thz_log = dus_thz_log.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
dus_soil_log = dus_soil_log.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
####RAW#########
dsoyb_d_raw = pd.concat([ds0_raw, dus_mst_raw, dus_thz_raw, dus_soil_raw], axis='columns')
######LOG#########
dsoyb_d = pd.concat([ds0_log, dus_mst_log, dus_thz_log, dus_soil_log], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
#####RAW#####
dsoyb_dus_raw = dsoyb_d_raw.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')
########LOG#######
dsoyb_dus_log = dsoyb_d.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dsoyb_val_raw = dsoyb_dus_raw.sample(frac=0.2, random_state=2705) #RAW
dsoyb_val_log = dsoyb_dus_log.sample(frac=0.2, random_state=2705) #LOG
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dsoyb_fit_raw = dsoyb_dus_raw.drop(dsoyb_val_raw.index) #RAW
dsoyb_fit_log = dsoyb_dus_log.drop(dsoyb_val_log.index) #LOG

##################Collinearity################################

###########RAW#################

grid = sb.PairGrid(data= dsoyb_fit_raw,
                    vars = ['n_fertilizer', 'p_fertilizer', 'n_total',
                    'pesticides_H', 'mechanized', 'irrigation'], height = 4)
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
                     edgecolor = 'k')
grid = grid.map_lower(sb.kdeplot, cmap = 'Reds')
#wanted to display the correlation coefficient in the lower triangle but don't know how
#grid = grid.map_lower(corr)

sb.pairplot(dsoyb_dus_raw)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dsoyb_cor_raw = dsoyb_fit_raw.drop(['lat', 'lon', 'area', 'yield'], axis='columns')
#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(dsoyb_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp = dsoyb_cor_raw.corr(method='spearman')
print(sp)
sp.iloc[0,1:5]
sp.iloc[1,2:5]
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

X = add_constant(dsoyb_cor_raw)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
#drop separate n variables
cor_n_total_raw = dsoyb_cor_raw.drop(['n_fertilizer', 'n_manure', 'p_fertilizer', 'S4_moderate_lim', 'Trop_low'], axis='columns')
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
mod = smf.ols(formula=' yield ~ n_total + pesticides_H + mechanized + irrigation', data=dsoyb_fit_raw)

mod = smf.ols(formula='yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dsoyb_fit_raw)

#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dsoyb_fit_raw, return_type='dataframe')


#define x and y dataframes
#Y containing only yield
mop = ds0_raw.iloc[:,3]
m_endog_raw = dsoyb_fit_raw.iloc[:,3] #RAW
m_endog_log = dsoyb_fit_log.iloc[:,3] #LOG
#X containing all variables
m_exog = ds0_raw.iloc[:,4]
m_exog_alln_raw = dsoyb_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #RAW
m_exog_alln_log = dsoyb_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #LOG
#test with n total and p
m_exog_np_raw = dsoyb_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns') #RAW
m_exog_np_log = dsoyb_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns')  #LOG
#test with n total without p
m_exog_n_log = dsoyb_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #RAW
m_exog_n_raw = dsoyb_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #LOG
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
mech = dsoyb_nozero['mechanized']
mech_total = mech.sum()
nonmech = dsoyb_nozero['non-mechanized']
non_mech_total = nonmech.sum()
total = mech_total + non_mech_total
mech_per = mech_total / total * 100
non_mech_per = non_mech_total / total * 100


#einfach nur für die iloc sachen drin
#drop lat, lon and area from the dataframe to only include relevant variables
dsoyb_rg = dsoyb_fit.iloc[:,[3,4,5,7,8,9,10]]
dsoyb_pl = dsoyb_fit.iloc[:,[4,5,7,8,9,10]]
dsoyb_yield = dsoyb_fit.iloc[:,3]

mod1 =sm.GLM(dsoyb_yield, dsoyb_pl, family=sm.families.Gamma())
#for some reason it works with Gaussian and Tweedie but not with Gamma or Inverse Gaussian... I really don't know why
mod_results = mod1.fit()
mod_res_alln_log = mod2.fit(method='qr')
    
'''
 


#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dsoyb_rg, return_type='dataframe')


