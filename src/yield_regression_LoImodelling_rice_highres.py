'''

File containing the code to explore data and perform a multiple regression
on yield for rice
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

#import yield geopandas data for rice

rice_yield=pd.read_csv(params.geopandasDataDir + 'RICECropYieldHighRes.csv')

#display first 5 rows of rice yield dataset
rice_yield.head()

#select all rows from rice_yield for which the column growArea has a value greater than zero
rice_nozero=rice_yield.loc[rice_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
rice_kgha=rice_nozero['yield_kgPerHa']
rice_kgha=rice_kgha.loc[rice_kgha < 17000]
#calculate descriptive statistics values (mean, median, standard deviation and variance)
#for the yield data with a value greater 0
rmean=rice_kgha.mean()
rmeadian=rice_kgha.median()
rsd=rice_kgha.std()
rvar=rice_kgha.var()
rmax=rice_kgha.max()
#calculate the mean with total production and area to check if the computed means align
rmean_total = ( rice_nozero['totalYield'].sum()) / (rice_nozero['growArea'].sum())
#the means do not align, probably due to the rebinning process
#calculate weighted mean (by area) of the yield colum
rmean_weighted = round(np.average(rice_kgha, weights=rice_nozero['growArea']),2)
#real weighted mean: 4373.68 kg/ha
#now they align!

#check the datatype of yield_kgPerHa and logarithmize the values
#logging is done to check the histogram and regoression fit of the transformed values
rice_kgha.dtype
rice_kgha_log=np.log(rice_kgha)

#plot rice yield distribution in a histogram
plt.hist(rice_kgha, bins=50)
plt.title('rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yield_kgPerHa
plt.hist(rice_kgha_log, bins=50)

#test if area without zeros aligns with FAOSTAT harvested area
rice_area_ha = sum(rice_nozero['growArea'])
print(rice_area_ha)
#164569574.0937798
#164586904	#FAOSTAT area from 2010 for rice

#subplot for all histograms
fig, axs = plt.subplots(1, 2, figsize=(5, 5))
axs[0].hist(rice_kgha, bins=50)
axs[1].hist(rice_kgha_log, bins=50)


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
pdf_listr = []
dist_listr = []
param_dictr ={"Values":[]}
#set xr to bins in the range of the raw data
xr = np.linspace(0.01,
                25500, 100)

###################################################################################
#####Testing of multiple distributions visually and using logLik, AIC and BIC######
###################################################################################

#Exponential distribution
dist_listr.append('exponential')
#get parameters and store them in the dictionary
param1 = stats.expon.fit(rice_kgha)
param_dictr["Values"].append(param1)
print(param1)
#calculate pdf
pdf_fitted1 = stats.expon.pdf(xr, *param1)
#calculate log pdf and store it in the list
pdf_fitted_log1 = stats.expon.logpdf(rice_kgha, *param1)
pdf_listr.append(pdf_fitted_log1)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(xr, pdf_fitted1, lw=2, label="Fitted Exponential distribution")
plt.legend()
plt.show()

#Normal distribution
dist_listr.append('normal')
#get parameters and store them in the dictionary
param2 = stats.norm.fit(rice_kgha)
#param_list.append(param2)
param_dictr["Values"].append(param2)
print(param2)
#calculate pdf
pdf_fitted2 = stats.norm.pdf(xr, *param2)
#calculate log pdf and store it in the list
pdf_fitted_log2 = stats.norm.logpdf(rice_kgha, *param2)
pdf_listr.append(pdf_fitted_log2)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(xr, pdf_fitted2, lw=2, label="Fitted normal distribution")
plt.legend()
plt.show()

#Halfnorm distribution
dist_listr.append('halfnormal')
#get parameters and store them in the dictionary
param3 = stats.halfnorm.fit(rice_kgha)
#param_list.append(param3)
param_dictr["Values"].append(param3)
print(param3)
#calculate pdf
pdf_fitted3 = stats.halfnorm.pdf(xr, *param3)
#calculate log pdf and store it in the list
pdf_fitted_log3 = stats.halfnorm.logpdf(rice_kgha, *param3)
pdf_listr.append(pdf_fitted_log3)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(xr, pdf_fitted3, lw=2, label="Fitted halfnormal distribution")
plt.legend()
plt.show()


xr2 = np.linspace(0.01,
                17000, 100)
#Gamma distribution
dist_listr.append('Gamma')
#get parameters and store them in the dictionary
param4 = stats.gamma.fit(rice_kgha)
#param_list.append(param4)
param_dictr["Values"].append(param4)
print(param4)
#calculate pdf
pdf_fitted4 = stats.gamma.pdf(xr2, *param4)
#calculate log pdf and store it in the list
pdf_fitted_log4 = stats.gamma.logpdf(rice_kgha, *param4)
pdf_listr.append(pdf_fitted_log4)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(xr2, pdf_fitted4, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()

#Inverse Gamma distribution
dist_listr.append('Inverse Gamma')
#get parameters and store them in the dictionary
param5 = stats.invgamma.fit(rice_kgha)
#param_list.append(param5)
param_dictr["Values"].append(param5)
print(param5)
#calculate pdf
pdf_fitted5 = stats.invgamma.pdf(xr, *param5)
#calculate log pdf and store it in the list
pdf_fitted_log5 = stats.invgamma.logpdf(rice_kgha, *param5)
pdf_listr.append(pdf_fitted_log5)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(xr, pdf_fitted5, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.show()

xr1 = np.linspace(4,
                11, 100)
#Normal distribution on log values
dist_listr.append('normal on log')
#get parameters and store them in the dictionary
param6 = stats.norm.fit(rice_kgha_log)
#param_list.append(param2)
param_dictr["Values"].append(param6)
print(param6)
#calculate pdf
pdf_fitted6 = stats.norm.pdf(xr1, *param6)
#calculate log pdf and store it in the list
pdf_fitted_log6 = stats.norm.logpdf(rice_kgha_log, *param6)
pdf_listr.append(pdf_fitted_log6)
#plot data histogram and pdf curve
h = plt.hist(rice_kgha_log, bins=50, density=True)
plt.plot(xr1, pdf_fitted6, lw=2, label="Fitted normal distribution on log")
plt.legend()
plt.title('log rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.show()

#one in all plot
h = plt.hist(rice_kgha, bins=50, density=True)
plt.plot(xr, pdf_fitted1, lw=2, label="Fitted Exponential distribution")
plt.plot(xr, pdf_fitted2, lw=2, label="Fitted Normal distribution")
plt.plot(xr, pdf_fitted3, lw=2, label="Fitted Halfnormal distribution")
plt.plot(xr, pdf_fitted4, lw=2, label="Fitted Gamma distribution")
plt.plot(xr, pdf_fitted5, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.title('rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=20000)
plt.show()


#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_listr, pdf_listr, param_dictr)
'''
       Distribution  loglikelihood           AIC           BIC
7  normal on log  -3.743533e+05  7.487227e+05  7.488078e+05
6  Inverse Gamma  -2.782122e+06  5.564261e+06  5.564346e+06
4     halfnormal  -2.785834e+06  5.571683e+06  5.571768e+06
1    exponential  -2.800599e+06  5.601214e+06  5.601299e+06
3         normal  -2.843839e+06  5.687694e+06  5.687779e+06
5          Gamma  -3.058454e+06  6.116923e+06  6.117008e+06
0        lognorm  -3.407763e+06  6.815542e+06  6.815627e+06
2        weibull  -3.503202e+06  7.006420e+06  7.006505e+06
#best fit so far: normal on log values by far, then Inverse Gamma on non-log
'''

'''
Load factor data and extract zeros
'''
r_pesticides=pd.read_csv(params.geopandasDataDir + 'RicePesticidesHighRes.csv')
print(r_pesticides.columns)
print(r_pesticides.head())
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
print(r_pesticides.loc[6040])
print(fertilizer.loc[6040])
print(fertilizer_man.loc[6040])
print(irr_t.loc[6040])
print(tillage.loc[6040])
print(aez.loc[6040])
print(rice_yield.loc[6040])

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
print(rice_yield.columns.tolist())
l = rice_yield.loc[:,'lats']
'''

#################################################################################
##############Loading variables without log to test the effect###################
#################################################################################
data_raw = {"lat": rice_yield.loc[:,'lats'],
		"lon": rice_yield.loc[:,'lons'],
		"area": rice_yield.loc[:,'growArea'],
        "Y": rice_yield.loc[:,'yield_kgPerHa'],
		"n_fertilizer": fertilizer.loc[:,'n_kgha'],
		"p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_man_prod" : fertilizer_man.loc[:,'produced_kgha'],
        "n_total" : N_total,
        "pesticides_H": r_pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'is_mech'],
        "irrigation_tot": irr_tot,
        "irrigation_rel": irr_rel.loc[:,'frac_reliant'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
drice_raw = pd.DataFrame(data=data_raw)
#select only the rows where the area of the cropland is larger than 0
dr0_raw=drice_raw.loc[drice_raw['area'] > 0]

dr0_raw['pesticides_H'] = dr0_raw['pesticides_H'].replace(np.nan, -9)
dr0_raw['irrigation_rel'] = dr0_raw['irrigation_rel'].replace(np.nan, -9)

#test if there are cells with 0s for the AEZ classes (there shouldn't be any)
r_testt = dr0_raw.loc[dr0_raw['thz_class'] == 0] #only 82 0s
r_testm = dr0_raw.loc[dr0_raw['mst_class'] == 0] #only 82 0s
r_tests = dr0_raw.loc[dr0_raw['soil_class'] == 0]
#3201 0s probably due to the original soil dataset being in 30 arcsec resolution:
    #land/ocean boundaries, especially of islands, don't always align perfectly

#test if certain classes of the AEZ aren't present in the dataset because they
#represent conditions which aren't beneficial for plant growth
#thz_class: test Arctic and Bor_cold_with_permafrost
r_test_t9 = dr0_raw.loc[dr0_raw['thz_class'] == 9]
#14 with Boreal and permafrost: reasonable
r_test_t10 = dr0_raw.loc[dr0_raw['thz_class'] == 10]
#196 with Arctic: is reasonable

#mst_class: test LPG<60days
r_test_m = dr0_raw.loc[dr0_raw['mst_class'] == 1]
#8202 in LPG<60 days class: probably due to irrigation

#soil class: test urban, water bodies and very steep class
r_test_s1 = dr0_raw.loc[dr0_raw['soil_class'] == 1]
#9881 in very steep class: makes sense, there is marginal agriculture in
#agricultural outskirts
r_test_s7 = dr0_raw.loc[dr0_raw['soil_class'] == 7]
#1841 in water class: this doesn't make sense but also due to resolution
#I think these should be substituted
r_test_s8 = dr0_raw.loc[dr0_raw['soil_class'] == 8]
#1687 in urban class: probably due to finer resolution in soil class, e.g. course of 
#the Nile is completely classified with yield estimates even though there are many urban areas
#Question: should the urban datapoints be taken out due to them being unreasonable? But then again
#the other datasets most likely contain values in these spots as well (equally unprecise), so I would
#just lose information
#I could substitute them like the water bodies

#test mech dataset values
r_test_mech0 = dr0_raw.loc[dr0_raw['mechanized'] == 0] #134976
r_test_mech1 = dr0_raw.loc[dr0_raw['mechanized'] == 1] #132231
r_test_mechn = dr0_raw.loc[dr0_raw['mechanized'] == -9] #41358
#this is a problem: -9 is used as NaN value and there are way, way too many

r_test_f = dr0_raw.loc[dr0_raw['n_fertilizer'] < 0] #15974 0s, 7040 NaNs
r_test_pf = dr0_raw.loc[dr0_raw['p_fertilizer'] < 0] #20070 0s, 7040 NaNs 
r_test_man = dr0_raw.loc[dr0_raw['n_manure'] < 0] #17699 0s, no NaNs
r_test_p = dr0_raw.loc[dr0_raw['pesticides_H'] < 0] #no 0s, 130979 NaNs

dr0_raw['thz_class'] = dr0_raw['thz_class'].replace(0,np.nan)
dr0_raw['mst_class'] = dr0_raw['mst_class'].replace(0,np.nan)
dr0_raw['soil_class'] = dr0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
dr0_raw['thz_class'] = dr0_raw['thz_class'].replace([9,10],8)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
dr0_raw = dr0_raw.fillna(method='ffill')

#Handle the data by eliminating the rows without data:
dr0_elim = dr0_raw.loc[dr0_raw['pesticides_H'] > -9]
dr0_elim = dr0_elim.loc[dr0_raw['mechanized'] > -9] 

est_mechn = dr0_elim.loc[dr0_elim['n_fertilizer'] < 0] #98798

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
dr0_elim.loc[dr0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
dr0_elim.loc[dr0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
dr0_elim = dr0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
dr0_elim.loc[dr0_elim['n_total'] < 0, 'n_total'] = dr0_elim['n_fertilizer'] + dr0_elim['n_manure']

plt.hist(dr0_elim['mst_class'])

###############################################################################
##################################Outliers####################################
##############################################################################

r_out_f = dr0_elim.loc[dr0_elim['n_fertilizer'] > 400] #only 36 left
r_out_p = dr0_elim.loc[dr0_elim['p_fertilizer'] > 100] #36
r_out_man = dr0_elim.loc[dr0_elim['n_manure'] > 250] #9
r_out_prod = dr0_elim.loc[dr0_elim['n_man_prod'] > 1000] #9
r_out_n = dr0_elim.loc[(dr0_elim['n_manure'] > 250) | (dr0_elim['n_fertilizer'] > 400)] #has to be 78+35-1=112

r_mman = dr0_elim['n_manure'].mean() #5.046215311442884
r_medman = dr0_elim['n_manure'].median() #2.8063108825683596

print(dr0_raw['Y'].median())
dr0_qt = dr0_elim.quantile([.1, .25, .5, .75, .8, .95, .999,.9999])
dr0_qt = dr0_raw.quantile([.999,.9999])
dr0_qt.reset_index(inplace=True, drop=True)
dr0_yqt = dr0_elim.loc[dr0_elim['Y'] > dr0_qt.iloc[0,3]]
163504*0.001

#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('dr0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=dr0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=dr0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=dr0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=dr0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=dr0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=dr0_elim, x='Y')

ax = sb.boxplot(x=dr0_elim["Y"], orient='v')
ax = sb.boxplot(x=dr0_elim["n_fertilizer"])
ax = sb.boxplot(x=dr0_elim["p_fertilizer"])
ax = sb.boxplot(x=dr0_elim["n_manure"])
ax = sb.boxplot(x=dr0_elim["n_total"])
ax = sb.boxplot(x=dr0_elim["pesticides_H"])
ax = sb.boxplot(x=dr0_elim["irrigation_tot"])
ax = sb.boxplot(x=dr0_elim["irrigation_rel"])
ax = sb.boxplot(x="mechanized", y='Y', data=dr0_elim)
ax = sb.boxplot(x="thz_class", y='Y', hue='mechanized', data=dr0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=dr0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=dr0_elim)

#replace nonsense values in fertilizer and manure datasets
dr0_elim.loc[dr0_elim['n_fertilizer'] > 400, 'n_fertilizer'] = np.nan
dr0_elim.loc[dr0_elim['p_fertilizer'] > 100, 'p_fertilizer'] = np.nan
dr0_elim.loc[dr0_elim['n_manure'] > 250, 'n_manure'] = np.nan
#dr0_elim.loc[dr0_elim['n_man_prod'] > 1000, 'n_man_prod'] = np.nan
dr0_elim = dr0_elim.fillna(method='ffill')
dr0_elim['n_total'] = dr0_elim['n_manure'] + dr0_elim['n_fertilizer']

###############################################################################
############Loading log transformed values for all variables##################
##############################################################################


#using log values for the input into the regression
#unfortunately the ln of 0 is not defined
#just keeping the 0 would skew the results as that would imply a 1 in the data when there is a 0
#could just use the smallest value of the dataset as a substitute?
data_log = {"lat": rice_yield.loc[:,'lats'],
		"lon": rice_yield.loc[:,'lons'],
		"area": rice_yield.loc[:,'growArea'],
        "yield": np.log(rice_yield.loc[:,'yield_kgPerHa']),
		"n_fertilizer": np.log(fertilizer.loc[:,'n_kgha']),
		"p_fertilizer": np.log(fertilizer.loc[:,'p_kgha']),
        "n_manure": np.log(fertilizer_man.loc[:,'applied_kgha']),
        "n_total" : np.log(N_total),
        "pesticides_H": np.log(r_pesticides.loc[:,'total_H']),
        "mechanized": r_tillage.loc[:,'rice_is_mech'],
#        "irrigation": np.log(irrigation.loc[:,'area']),
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}


drice_log = pd.DataFrame(data=data_log)
#select all rows from drice_log for which the column growArea has a value greater than zero
dr0_log=drice_log.loc[drice_log['area'] > 0]
#the data contains -inf values because the n+p+pests+irrigation columns contain 0 values for which ln(x) is not defined 
#calculate the minimum values for each column disregarding -inf values to see which is the lowest value in the dataset (excluding lat & lon)
min_dr0_log=dr0_log[dr0_log.iloc[:,3:11]>-inf].min()
#replace the -inf values with the minimum of the dataset + 5 : this is done to achieve a distinction between very small
#values and actual zeros
dr0_log.replace(-inf, -30, inplace=True)
#check distribution of AEZ factors in the historgrams
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

plt.hist(dr0_log['soil_class'], bins=50)
plt.hist(dr0_log['mst_class'], bins=50)
plt.hist(dr0_log['thz_class'], bins=50)
#ONLY RUN THIS BLOCK WHEN WORKING AT LOW RESOLUTION!
#AEZ factors contain unexpected 0s due to resolution rebinning
#urban class is missing in soil because of rebinning (urban class to small to dominant a large cell)
#convert 0s in the AEZ columns to NaN values so that they can be replaced by the ffill method
#0s make no sense in the dataset that is limited to rice cropping area because the area is obviously on land
dr0_log['thz_class'] = dr0_log['thz_class'].replace(0,np.nan)
dr0_log['mst_class'] = dr0_log['mst_class'].replace(0,np.nan)
dr0_log['soil_class'] = dr0_log['soil_class'].replace(0,np.nan)
#NaN values throw errors in the regression, they need to be handled beforehand
#fill in the NaN vlaues in the dataset with a forward filling method (replacing NaN with the value in the cell before)
dr0_log = dr0_log.fillna(method='ffill')
#fill in the remaining couple of nans at the top of mechanized column
dr0_log['mechanized'] = dr0_log['mechanized'].fillna(1)

#Just some PLOTS

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot the continuous variables to get a sense of their distribution #RAW
plt.hist(dr0_raw['n_fertilizer'], bins=50)
plt.hist(dr0_raw['p_fertilizer'], bins=50)
plt.hist(dr0_raw['n_total'], bins=50)
plt.hist(dr0_raw['pesticides_H'], bins=100)
plt.hist(dr0_raw['irrigation'], bins=50)
'''
plt.ylim(0,5000)
plt.xlim(0, 0.04)
plt.title('rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
'''

#scatterplots for #RAW variables

dr0_raw.plot.scatter(x = 'n_fertilizer', y = 'yield')
dr0_raw.plot.scatter(x = 'p_fertilizer', y = 'yield')
dr0_raw.plot.scatter(x = 'pesticides_H', y = 'yield')
dr0_raw.plot.scatter(x = 'mechanized', y = 'yield')
dr0_raw.plot.scatter(x = 'non-mechanized', y = 'yield')
dr0_raw.plot.scatter(x = 'irrigation', y = 'yield')

#scatterplots and histograms for #LOG variables
dr0_log.plot.scatter(x = 'n_fertilizer', y = 'yield')
dr0_log.plot.scatter(x = 'p_fertilizer', y = 'yield')
dr0_log.plot.scatter(x = 'pesticides_H', y = 'yield')
dr0_log.plot.scatter(x = 'mechanized', y = 'yield')
dr0_log.plot.scatter(x = 'n_total', y = 'yield')
dr0_log.plot.scatter(x = 'irrigation', y = 'yield')
dr0_log.plot.scatter(x = 'thz_class', y = 'yield')
dr0_log.plot.scatter(x = 'mst_class', y = 'yield')
dr0_log.plot.scatter(x = 'soil_class', y = 'yield')

plt.hist(dr0_log['n_fertilizer'], bins=50)
plt.hist(dr0_log['p_fertilizer'], bins=50)
plt.hist(dr0_log['n_total'], bins=50)
plt.hist(dr0_log['pesticides_H'], bins=100)
plt.hist(dr0_log['irrigation'], bins=50)
plt.hist(dr0_log['mechanized'], bins=50)
plt.hist(dr0_log['thz_class'], bins=50)
plt.hist(dr0_log['mst_class'], bins=50)
plt.hist(dr0_log['soil_class'], bins=50)
plt.ylim(0,5000)
plt.title('rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
#####RAW##########
dur_mst_raw = pd.get_dummies(dr0_raw['mst_class'])
dur_thz_raw = pd.get_dummies(dr0_raw['thz_class'])
dur_soil_raw = pd.get_dummies(dr0_raw['soil_class'])
#####LOG##########
dur_mst_log = pd.get_dummies(dr0_log['mst_class'])
dur_thz_log = pd.get_dummies(dr0_log['thz_class'])
dur_soil_log = pd.get_dummies(dr0_log['soil_class'])
#rename the columns according to the classes
#####RAW##########
dur_mst_raw = dur_mst_raw.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
dur_thz_raw = dur_thz_raw.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
dur_soil_raw = dur_soil_raw.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#######LOG#########
dur_mst_log = dur_mst_log.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
dur_thz_log = dur_thz_log.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor_cold_noPFR", 
                        9:"Bor_cold_PFR", 10:"Arctic"}, errors="raise")
dur_soil_log = dur_soil_log.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr", 7:"L2_water"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
####RAW#########
drice_d_raw = pd.concat([dr0_raw, dur_mst_raw, dur_thz_raw, dur_soil_raw], axis='columns')
######LOG#########
drice_d = pd.concat([dr0_log, dur_mst_log, dur_thz_log, dur_soil_log], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
#####RAW#####
drice_dum_raw = drice_d_raw.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')
########LOG#######
drice_dum_log = drice_d.drop(['mst_class', 'thz_class', 'soil_class', 'LGP<60days', 
                      'Arctic', 'L2_water'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
drice_val_raw = drice_dum_raw.sample(frac=0.2, random_state=2705) #RAW
drice_val_log = drice_dum_log.sample(frac=0.2, random_state=2705) #LOG
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
drice_fit_raw = drice_dum_raw.drop(drice_val_raw.index) #RAW
drice_fit_log = drice_dum_log.drop(drice_val_log.index) #LOG

##################Collinearity################################

###########RAW#################

grid = sb.PairGrid(data= drice_fit_raw,
                    vars = ['n_fertilizer', 'p_fertilizer', 'n_total',
                    'pesticides_H', 'mechanized', 'irrigation'], height = 4)
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
                     edgecolor = 'k')
grid = grid.map_lower(sb.kdeplot, cmap = 'Reds')
#wanted to display the correlation coefficient in the lower triangle but don't know how
#grid = grid.map_lower(corr)

sb.pairplot(drice_dum_raw)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
drice_cor_raw = drice_fit_raw.drop(['lat', 'lon', 'area', 'yield'], axis='columns')
#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(drice_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp = drice_cor_raw.corr(method='spearman')
print(sp)
sp.iloc[0,1:5]
sp.iloc[1,2:5]
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

X = add_constant(drice_cor_elim)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
'''
const                328.358538
p_fertilizer           6.140546
n_total                5.846132
pesticides_H           2.272387
mechanized             1.342700
irrigation_tot         1.857810
LGP<60days             2.334439
60-120days             3.651814
120-180days            7.431414
180-225days            7.002788
225-270days            7.775257
270-365days           11.823382
Trop_low              67.483514
Trop_high              6.937229
Sub-trop_warm         35.702874
Sub-trop_mod_cool     28.018553
Sub-trop_cool         14.587196
Temp_mod              20.954271
Temp_cool             10.144699
S1_very_steep          1.406882
S2_hydro_soil          1.213223
S3_no-slight_lim       2.180892
S4_moderate_lim        3.037877
S5_severe_lim          1.807235
dtype: float64
'''

######################TEST#########################

test_R = dr0_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel'], axis='columns')
test_R['thz_class'] = test_R['thz_class'].replace([8],7)

test_R['mst_class'] = test_R['mst_class'].replace([2],1)
test_R['mst_class'] = test_R['mst_class'].replace([7],6)

plt.hist(dr0_elim['soil_class'])
bor_test = dr0_elim.loc[dr0_elim['thz_class'] == 8] #614

rd_mst = pd.get_dummies(test_R['mst_class'])
rd_thz = pd.get_dummies(test_R['thz_class'])

rd_mst = rd_mst.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
rd_thz = rd_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
test_R = pd.concat([test_R, rd_mst, rd_thz, dur_soil_elim], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
test_R.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns', inplace=True)

test_cor_elim = test_R.drop(['thz_class','mst_class', 'soil_class'], axis='columns')

#drop dummy variables
cor_test = test_cor_elim.loc[:,['n_manure', 'mechanized', 'thz_class', 'mst_class', 
                                   'soil_class']]
X2 = add_constant(test_cor_elim)
pd.Series([variance_inflation_factor(X2.values, i) 
               for i in range(X2.shape[1])], 
              index=X2.columns)

plt.hist(test_R['mst_class'], bins=50)
ax = sb.boxplot(x=test_R["mst_class"], y=dr0_elim['Y'])
plt.ylim(0,20000)

'''
mst_test

const                294.210696
p_fertilizer           6.077448
n_total                5.774830
pesticides_H           2.279151
mechanized             1.330683
irrigation_tot         1.855685
LGP<120days            1.387747
120-180days            1.482771
180-225days            1.342825
225-270days            1.271386
Trop_low              68.502526
Trop_high              7.046224
Sub-trop_warm         36.221347
Sub-trop_mod_cool     28.506731
Sub-trop_cool         14.763804
Temp_mod              21.244114
Temp_cool             10.371542
S1_very_steep          1.402698
S2_hydro_soil          1.211018
S3_no-slight_lim       2.168223
S4_moderate_lim        3.026065
S5_severe_lim          1.804808
dtype: float64

thz_test

const                53.903056
p_fertilizer          6.070424
n_total               5.773618
pesticides_H          2.277836
mechanized            1.324546
irrigation_tot        1.849693
LGP<120days           1.386966
120-180days           1.480541
180-225days           1.339101
225-270days           1.270193
Trop_low              8.397431
Trop_high             1.643247
Sub-trop_warm         4.974006
Sub-trop_mod_cool     3.708061
Sub-trop_cool         2.430022
Temp_mod              2.967455
S1_very_steep         1.388197
S2_hydro_soil         1.210928
S3_no-slight_lim      2.167974
S4_moderate_lim       3.016111
S5_severe_lim         1.800800
dtype: float64

'''



######################Regression##############################

#R-style formula
#doesn't work for some reason... I always get parsing errors and I don't know why
mod = smf.ols(formula=' yield ~ n_total + pesticides_H + mechanized + irrigation', data=drice_fit_raw)

mod = smf.ols(formula='yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=drice_fit_raw)

#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=drice_fit_raw, return_type='dataframe')


#define x and y dataframes
#Y containing only yield
mop = dr0_raw.iloc[:,3]
m_endog_raw = drice_fit_raw.iloc[:,3] #RAW
m_endog_log = drice_fit_log.iloc[:,3] #LOG
#X containing all variables
m_exog = dr0_raw.iloc[:,4]
m_exog_alln_raw = drice_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #RAW
m_exog_alln_log = drice_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_total'], axis='columns') #LOG
#test with n total and p
m_exog_np_raw = drice_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns') #RAW
m_exog_np_log = drice_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure'], axis='columns')  #LOG
#test with n total without p
m_exog_n_log = drice_fit_raw.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #RAW
m_exog_n_raw = drice_fit_log.drop(['yield', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer'], axis='columns') #LOG
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
mech = drice_nozero['mechanized']
mech_total = mech.sum()
nonmech = drice_nozero['non-mechanized']
non_mech_total = nonmech.sum()
total = mech_total + non_mech_total
mech_per = mech_total / total * 100
non_mech_per = non_mech_total / total * 100


#einfach nur für die iloc sachen drin
#drop lat, lon and area from the dataframe to only include relevant variables
drice_rg = drice_fit.iloc[:,[3,4,5,7,8,9,10]]
drice_pl = drice_fit.iloc[:,[4,5,7,8,9,10]]
drice_yield = drice_fit.iloc[:,3]

mod1 =sm.GLM(drice_yield, drice_pl, family=sm.families.Gamma())
#for some reason it works with Gaussian and Tweedie but not with Gamma or Inverse Gaussian... I really don't know why
mod_results = mod1.fit()
mod_res_alln_log = mod2.fit(method='qr')
    
'''
 


#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=drice_rg, return_type='dataframe')


