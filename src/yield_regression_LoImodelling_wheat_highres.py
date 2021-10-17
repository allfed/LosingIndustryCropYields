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
wheat_kgha = wheat_kgha.loc[wheat_kgha < 14000]

wheat_nozero['growArea'].min() #0.1 ha
wheat_nozero['growArea'].max() #10253.6 ha
wheat_nozero['growArea'].mean() #477.71 ha
tt3 = (wheat_nozero['yield_kgPerHa'] * wheat_nozero['growArea']).sum()
ar_t = wheat_nozero.loc[wheat_nozero['growArea'] < 10] #99701 cells ~21.58%
ar_t1 = wheat_nozero.loc[wheat_nozero['growArea'] > 1000] #72408 cells ~15.67% but ~64.61% of the yield...
tt = (ar_t1['yield_kgPerHa'] * ar_t1['growArea']).sum() #436049166701.7425 kg
ar_t2 = wheat_nozero.loc[wheat_nozero['growArea'] > 100] #255326 cells ~55.26% but ~98.19% of the yield...
tt2 = (ar_t2['yield_kgPerHa'] * ar_t2['growArea']).sum()
662594058048.5144/674830448927.756 #
ax = sb.boxplot(x=ar_t2["growArea"])
255326/462033

'''
Attempts to weight the values (not successful so far)
wheat_kg = wheat_nozero['totalYield']
wweights=wheat_nozero['growArea']/wheat_nozero['growArea'].sum()
wweigh_sum = wweights.sum()
print(wheat_nozero['growArea'].max())
print(wheat_nozero['growArea'].min())
wheat_weighted=wheat_kgha*wweights*wheat_nozero['growArea'].sum()
print(wheat_weighted.mean())

#Plots: belong to the weighting attempts
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
'''
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
                14000, 100)

###################################################################################
#####Testing of multiple distributions visually and using logLik, AIC and BIC######
###################################################################################

#Lognormal distribution
dist_listw.append('lognorm')
#fit distribution to rice yield data to get values for the parameters
param1 = stats.invgauss.fit(wheat_kgha)
#store the parameters in the initialized dictionary
param_dictw["Values"].append(param1)
print(param1)
#use the parameters to calculate values for the probability density function 
#(pdf) of the distribution
pdf_fitted = stats.invgauss.pdf(xw, *param1)
#calculate the logarithmized pdf to calculate statistical values for the fit
pdf_fitted_log = stats.invgauss.logpdf(wheat_kgha, *param1)
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
w_pesticides=pd.read_csv(params.geopandasDataDir + 'WheatPesticidesHighRes.csv')
print(w_pesticides.columns)
print(w_pesticides.head())
fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv') #kg/m²
print(fertilizer.columns)
print(fertilizer.head())
fertilizer_man=pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv') #kg/km²
print(fertilizer_man.columns)
print(fertilizer_man.head())
irr_t=pd.read_csv(params.geopandasDataDir + 'FracIrrigationAreaHighRes.csv')
print(irr_t.columns)
print(irr_t.head())
crop = pd.read_csv(params.geopandasDataDir + 'FracCropAreaHighRes.csv')
irr_rel=pd.read_csv(params.geopandasDataDir + 'FracReliantHighRes.csv')
tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighResAllCrops.csv')
print(tillage.columns)
print(tillage.head())
aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')
print(aez.columns)
print(aez.head())
print(aez.dtypes)

#fraction of irrigation total is of total cell area so I have to divide it by the
#fraction of crop area in a cell and set all values >1 to 1
irr_tot = irr_t['fraction']/crop['fraction']
irr_tot.loc[irr_tot > 1] = 1
#dividing by 0 leaves a NaN value, so I have them all back to 0
irr_tot.loc[irr_tot.isna()] = 0

#print the value of each variable at the same index to make sure that coordinates align (they do)
print(w_pesticides.loc[1444612])
print(fertilizer.loc[1444612])
print(fertilizer_man.loc[1444612])
print(irr_t.loc[1444612])
print(tillage.loc[1444612])
print(aez.loc[1444612])
print(wheat_yield.loc[1444612])

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
dataw_raw = {"lat": wheat_yield.loc[:,'lats'],
		"lon": wheat_yield.loc[:,'lons'],
		"area": wheat_yield.loc[:,'growArea'],
        "Y": wheat_yield.loc[:,'yield_kgPerHa'],
		"n_fertilizer": fertilizer.loc[:,'n_kgha'],
		"p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_man_prod" : fertilizer_man.loc[:,'produced_kgha'],
        "n_total" : N_total,
        "pesticides_H": w_pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'is_mech'],
        "irrigation_tot": irr_tot,
        "irrigation_rel": irr_rel.loc[:,'frac_reliant'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
dwheat_raw = pd.DataFrame(data=dataw_raw)
#select only the rows where the area of the cropland is larger than 0
dw0_raw=dwheat_raw.loc[dwheat_raw['area'] > 0]

dw0_raw['pesticides_H'] = dw0_raw['pesticides_H'].replace(np.nan, -9)
dw0_raw['irrigation_rel'] = dw0_raw['irrigation_rel'].replace(np.nan, -9)

#test if there are cells with 0s for the AEZ classes (there shouldn't be any)
w_testt = dw0_raw.loc[dw0_raw['thz_class'] == 0] #only one 0
w_testm = dw0_raw.loc[dw0_raw['mst_class'] == 0] #only one 0
w_tests = dw0_raw.loc[dw0_raw['soil_class'] == 0]
#850 0s probably due to the original soil dataset being in 30 arcsec resolution:
    #land/ocean boundaries, especially of islands, don't always align perfectly

#test if certain classes of the AEZ aren't present in the dataset because they
#represent conditions which aren't beneficial for plant growth
#thz_class: test Arctic and Bor_cold_with_permafrost
w_test_t9 = dw0_raw.loc[dw0_raw['thz_class'] == 9]
#459 with Boreal and permafrost: reasonable
w_test_t10 = dw0_raw.loc[dw0_raw['thz_class'] == 10]
#168 with Arctic: is reasonable

#mst_class: test LPG<60days
w_test_m = dw0_raw.loc[dw0_raw['mst_class'] == 1]
#23892 in LPG<60 days class: probably due to irrigation

#soil class: test urban, water bodies and very steep class
w_test_s1 = dw0_raw.loc[dw0_raw['soil_class'] == 1]
#14593 in very steep class: makes sense, there is marginal agriculture in
#agricultural outskirts
w_test_s7 = dw0_raw.loc[dw0_raw['soil_class'] == 7]
#3033 in water class: this doesn't make sense but also due to resolution
#I think these should be substituted
w_test_s8 = dw0_raw.loc[dw0_raw['soil_class'] == 8]
#3663 in urban class: probably due to finer resolution in soil class, e.g. course of 
#the Nile is completely classified with yield estimates even though there are many urban areas
#Question: should the urban datapoints be taken out due to them being unreasonable? But then again
#the other datasets most likely contain values in these spots as well (equally unprecise), so I would
#just lose information
#I could substitute them like the water bodies

#test mech dataset values
w_test_mech0 = dw0_raw.loc[dw0_raw['mechanized'] == 0] #92565
w_test_mech1 = dw0_raw.loc[dw0_raw['mechanized'] == 1] #760670
w_test_mechn = dw0_raw.loc[dw0_raw['mechanized'] == -9] #98798
#this is a problem: -9 is used as NaN value and there are way, way too many

w_test_f = dw0_raw.loc[dw0_raw['n_fertilizer'] < 0] #19044 0s, 4512 NaNs
w_test_pf = dw0_raw.loc[dw0_raw['p_fertilizer'] < 0] #25889 0s, 4512 NaNs
w_test_man = dw0_raw.loc[dw0_raw['n_manure'] < 0] #12296 0s, 0 NaNs
w_test_p = dw0_raw.loc[dw0_raw['pesticides_H'].isna()] #no 0s, 120056 NaNs

dw0_raw['thz_class'] = dw0_raw['thz_class'].replace(0,np.nan)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(0,np.nan)
dw0_raw['soil_class'] = dw0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
dw0_raw['thz_class'] = dw0_raw['thz_class'].replace([9,10],8)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
dw0_raw = dw0_raw.fillna(method='ffill')

#Handle the data by eliminating the rows without data:
dw0_elim = dw0_raw.loc[dw0_raw['pesticides_H'] > -9]
dw0_elim = dw0_elim.loc[dw0_raw['mechanized'] > -9] 

est_mechn = dw0_elim.loc[dw0_elim['mechanized'] == -9] #98798

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
dw0_elim.loc[dw0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
dw0_elim.loc[dw0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
dw0_elim = dw0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
dw0_elim.loc[dw0_elim['n_total'] < 0, 'n_total'] = dw0_elim['n_fertilizer'] + dw0_elim['n_manure']

plt.hist(dw0_elim['soil_class'])

########################################################################
##############################Outliers##################################
########################################################################

w_out_f = dw0_elim.loc[dw0_elim['n_fertilizer'] > 400] #0
w_out_p = dw0_elim.loc[dw0_elim['p_fertilizer'] > 100] #400
w_out_man = dw0_elim.loc[dw0_elim['n_manure'] > 250] #28
w_out_prod = dw0_elim.loc[dw0_elim['n_man_prod'] > 1000] #16
w_out_n = dw0_elim.loc[(dw0_elim['n_manure'] > 250) | (dw0_elim['n_fertilizer'] > 400)] #has to be 78+35-1=112

w_mman = dw0_elim['n_manure'].mean() #5.543249878953387
w_medman = dw0_elim['n_manure'].median() #2.763350067138672

dw0_qt = dw0_elim.quantile([.1, .25, .5, .75, .8, .95, .999,.9999])
dw0_qt = dw0_raw.quantile([.999,.9999])
dw0_qt.reset_index(inplace=True, drop=True)
dw0_yqt = dw0_elim.loc[dw0_elim['Y'] > dw0_qt.iloc[0,3]]
163504*0.001


#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('dw0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=dw0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=dw0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=dw0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=dw0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=dw0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=dw0_elim, x='Y')

ax = sb.boxplot(x=dw0_elim["Y"], orient='v')
ax = sb.boxplot(x=dw0_elim["n_fertilizer"])
ax = sb.boxplot(x=dw0_elim["p_fertilizer"])
ax = sb.boxplot(x=dw0_elim["n_manure"])
ax = sb.boxplot(x=dw0_elim["n_total"])
ax = sb.boxplot(x=dw0_elim["pesticides_H"])
ax = sb.boxplot(x=dw0_elim["irrigation_tot"])
ax = sb.boxplot(x=dw0_elim["irrigation_rel"])
ax = sb.boxplot(x="mechanized", y='Y', data=dw0_elim)
ax = sb.boxplot(x="thz_class", y='Y', data=dw0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=dw0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=dw0_elim)
plt.ylim(0,20000)

#replace nonsense values in fertilizer and manure datasets
dw0_elim.loc[dw0_elim['n_fertilizer'] > 400, 'n_fertilizer'] = np.nan
dw0_elim.loc[dw0_elim['p_fertilizer'] > 100, 'p_fertilizer'] = np.nan
dw0_elim.loc[dw0_elim['n_manure'] > 250, 'n_manure'] = np.nan
#dw0_elim.loc[dw0_elim['n_man_prod'] > 1000, 'n_man_prod'] = np.nan
dw0_elim = dw0_elim.fillna(method='ffill')
dw0_elim['n_total'] = dw0_elim['n_manure'] + dw0_elim['n_fertilizer']


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
        "pesticides_H": np.log(w_pesticides.loc[:,'total_H']),
        "mechanized": w_tillage.loc[:,'whea_is_mech'],
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

X = add_constant(dwheat_cor_elim)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
'''
const                618.200415
p_fertilizer           5.144046
n_total                6.458572
pesticides_H           1.995414
mechanized             2.042481
irrigation_tot         2.040981
LGP<60days            19.495530
60-120days            66.122849
120-180days          107.508652
180-225days           83.373819
225-270days           66.015922
270-365days           78.714485
Trop_low               3.882418
Trop_high              3.002226
Sub-trop_warm          8.022661
Sub-trop_mod_cool     10.808561
Sub-trop_cool          9.378346
Temp_mod              10.195501
Temp_cool             17.624626
S1_very_steep          1.379885
S2_hydro_soil          1.361384
S3_no-slight_lim       3.735970
S4_moderate_lim        2.982573
S5_severe_lim          1.464532
dtype: float64
'''
######################TEST#########################

test_W = dw0_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel'], axis='columns')
test_W['thz_class'] = test_W['thz_class'].replace([8],7)

test_W['mst_class'] = test_W['mst_class'].replace([2],1)
test_W['mst_class'] = test_W['mst_class'].replace([7],6)

plt.hist(dw0_elim['soil_class'])
bor_test = dw0_elim.loc[dw0_elim['thz_class'] == 7] #3994

wd_mst = pd.get_dummies(test_W['mst_class'])
wd_thz = pd.get_dummies(test_W['thz_class'])

wd_mst = wd_mst.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
wd_thz = wd_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
test_W = pd.concat([test_W, wd_mst, wd_thz, duw_soil_elim], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
test_W.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns', inplace=True)

test_cor_elim = test_W.drop(['thz_class','mst_class', 'soil_class'], axis='columns')

#drop dummy variables
cor_test = test_cor_elim.loc[:,['n_manure', 'mechanized', 'thz_class', 'mst_class', 
                                   'soil_class']]
X2 = add_constant(test_cor_elim)
pd.Series([variance_inflation_factor(X2.values, i) 
               for i in range(X2.shape[1])], 
              index=X2.columns)

plt.hist(test_W['mst_class'], bins=50)
ax = sb.boxplot(x=test_W["mst_class"], y=dw0_elim['Y'])
plt.ylim(0,20000)

'''
mst_test

const                106.453734
p_fertilizer           5.037159
n_total                6.280262
pesticides_H           1.989920
mechanized             2.013866
irrigation_tot         2.032186
LGP<120days            2.195523
120-180days            2.323319
180-225days            1.984680
225-270days            1.668748
Trop_low               3.905866
Trop_high              3.006388
Sub-trop_warm          8.013512
Sub-trop_mod_cool     10.884383
Sub-trop_cool          9.417410
Temp_mod              10.269156
Temp_cool             17.776263
S1_very_steep          1.375200
S2_hydro_soil          1.362108
S3_no-slight_lim       3.726079
S4_moderate_lim        2.975657
S5_severe_lim          1.462976
dtype: float64

thz_test

const                38.730934
p_fertilizer          5.035914
n_total               6.269529
pesticides_H          1.987311
mechanized            2.011403
irrigation_tot        2.031749
LGP<120days           2.194730
120-180days           2.323044
180-225days           1.984607
225-270days           1.665849
Trop_low              1.313606
Trop_high             1.200082
Sub-trop_warm         1.797538
Sub-trop_mod_cool     1.481828
Sub-trop_cool         1.510802
Temp_mod              1.402745
S1_very_steep         1.374092
S2_hydro_soil         1.361691
S3_no-slight_lim      3.725888
S4_moderate_lim       2.973985
S5_severe_lim         1.462609
dtype: float64

'''


###########LOG##################


######################Regression##############################

#R-style formula
#doesn't work for some reason... I always get parsing errors and I don't know why
#determine models
#Normal distribution
w_mod_elimn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dwheat_fit_elim)
#Gamma distribution
w_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dwheat_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
w_mod_elim0 = smf.glm(formula='Y ~ 1', data=dwheat_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
w_fit_elimn = w_mod_elimn.fit()
w_fit_elimg = w_mod_elimg.fit()
w_fit_elim0 = w_mod_elim0.fit()
#print results
print(w_fit_elimn.summary()) #0.335
#LogLik: -2021300; AIC: 4043000; BIC: 4043000
print(w_fit_elimg.summary())
print(w_fit_elim0.summary())


###########Fit statistics#############
#calculate pseudo R² for the Gamma distribution
w_pseudoR_elim = 1-(77588/112650) #0.31124
print(w_pseudoR_elim)

#calculate AIC and BIC for Gamma
w_aic = w_fit_elimg.aic 
w_bic = w_fit_elimg.bic_llf

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


