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

params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for maize

maize_yield=pd.read_csv(params.geopandasDataDir + 'MAIZCropYieldHighRes.csv')

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

maize_nozero['growArea'].min() #0.1 ha
maize_nozero['growArea'].max() #10945.4 ha
maize_nozero['growArea'].mean() #292.94 ha
tt3 = (maize_nozero['yield_kgPerHa'] * maize_nozero['growArea']).sum()
ar_t = maize_nozero.loc[maize_nozero['growArea'] < 10] #148655 cells ~26.46%
ar_t1 = maize_nozero.loc[maize_nozero['growArea'] > 1000] #44900 cells ~7.99% but ~61.99% of the yield...
tt = (ar_t1['yield_kgPerHa'] * ar_t1['growArea']).sum()
ar_t2 = maize_nozero.loc[maize_nozero['growArea'] > 100] #235669 cells ~41.95% but ~96.76% of the yield...
tt2 = (ar_t1['yield_kgPerHa'] * ar_t1['growArea']).sum()
825496849007.0315/853148561445.0236 #
ax = sb.boxplot(x=maize_nozero["growArea"])
235669/561780
#check the datatype of yield_kgPerHa and logarithmize the values
#logging is done to check the histogram and regoression fit of the transformed values
maize_kgha.dtype
maize_kgha_log=np.log(maize_kgha)

plt.hist(ar_t1['yield_kgPerHa'], bins=50)
x = sb.boxplot(x=ar_t1["yield_kgPerHa"])
plt.scatter(ar_t1["growArea"], ar_t1["yield_kgPerHa"])

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
pdf_listm = []
dist_listm = []
param_dictm ={"Values":[]}
#set xm to bins in the range of the raw data
xm = np.linspace(0.01,
                11000, 100)

###################################################################################
#####Testing of multiple distributions visually and using logLik, AIC and BIC######
###################################################################################

#Lognormal distribution
dist_listm.append('lognorm')
#fit distribution to rice yield data to get values for the parameters
param1 = stats.lognorm.fit(maize_kgha)
#store the parameters in the initialized dictionary
param_dictm["Values"].append(param1)
print(param1)
#use the parameters to calculate values for the probability density function 
#(pdf) of the distribution
pdf_fitted = stats.lognorm.pdf(xm, *param1)
#calculate the logarithmized pdf to calculate statistical values for the fit
pdf_fitted_log = stats.lognorm.logpdf(maize_kgha, *param1)
#store the log pdf in the pdf list
pdf_listm.append(pdf_fitted_log)
#plot the histogram of the yield data and the curve of the lognorm pdf
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.legend()
plt.show()


#Exponential distribution
dist_listm.append('exponential')
#get parameters and store them in the dictionary
param2 = stats.expon.fit(maize_kgha)
param_dictm["Values"].append(param2)
print(param2)
#calculate pdf
pdf_fitted2 = stats.expon.pdf(xm, *param2)
#calculate log pdf and store it in the list
pdf_fitted_log2 = stats.expon.logpdf(maize_kgha, *param2)
pdf_listm.append(pdf_fitted_log2)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.legend()
plt.show()


#Weibull distribution
dist_listm.append('weibull')
#get parameters and store them in the dictionary
param3 = stats.weibull_min.fit(maize_kgha)
#param_list.append(param3)
param_dictm["Values"].append(param3)
print(param3)
#calculate pdf
pdf_fitted3 = stats.weibull_min.pdf(xm, *param3)
#calculate log pdf and store it in the list
pdf_fitted_log3 = stats.weibull_min.logpdf(maize_kgha, *param3)
pdf_listm.append(pdf_fitted_log3)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.ylim(top=0.00025)
plt.legend()
plt.show()

#Normal distribution
dist_listm.append('normal')
#get parameters and store them in the dictionary
param4 = stats.norm.fit(maize_kgha)
#param_list.append(param4)
param_dictm["Values"].append(param4)
print(param4)
#calculate pdf
pdf_fitted4 = stats.norm.pdf(xm, *param4)
#calculate log pdf and store it in the list
pdf_fitted_log4 = stats.norm.logpdf(maize_kgha, *param4)
pdf_listm.append(pdf_fitted_log4)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted4, lw=2, label="Fitted normal distribution")
plt.legend()
plt.show()

#Halfnorm distribution
dist_listm.append('halfnormal')
#get parameters and store them in the dictionary
param5 = stats.halfnorm.fit(maize_kgha)
#param_list.append(param5)
param_dictm["Values"].append(param5)
print(param5)
#calculate pdf
pdf_fitted5 = stats.halfnorm.pdf(xm, *param5)
#calculate log pdf and store it in the list
pdf_fitted_log5 = stats.halfnorm.logpdf(maize_kgha, *param5)
pdf_listm.append(pdf_fitted_log5)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted5, lw=2, label="Fitted halfnormal distribution")
plt.legend()
plt.show()

#Gamma distribution
dist_listm.append('Gamma')
#get parameters and store them in the dictionary
param6 = stats.gamma.fit(maize_kgha)
#param_list.append(param6)
param_dictm["Values"].append(param6)
print(param6)
#calculate pdf
pdf_fitted6 = stats.gamma.pdf(xm, *param6)
#calculate log pdf and store it in the list
pdf_fitted_log6 = stats.gamma.logpdf(maize_kgha, *param6)
pdf_listm.append(pdf_fitted_log6)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted6, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()

#Inverse Gamma distribution
dist_listm.append('Inverse Gamma')
#get parameters and store them in the dictionary
param7 = stats.invgamma.fit(maize_kgha)
#param_list.append(param5)
param_dictm["Values"].append(param7)
print(param7)
#calculate pdf
pdf_fitted7 = stats.invgamma.pdf(xm, *param7)
#calculate log pdf and store it in the list
pdf_fitted_log7 = stats.invgamma.logpdf(maize_kgha, *param7)
pdf_listm.append(pdf_fitted_log7)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted7, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.show()

xm1 = np.linspace(4,
                11, 100)
#Normal distribution on log values
dist_listm.append('normal on log')
#get parameters and store them in the dictionary
param8 = stats.norm.fit(maize_kgha_log)
#param_list.append(param4)
param_dictm["Values"].append(param8)
print(param8)
#calculate pdf
pdf_fitted8 = stats.norm.pdf(xm1, *param8)
#calculate log pdf and store it in the list
pdf_fitted_log8 = stats.norm.logpdf(maize_kgha_log, *param8)
pdf_listm.append(pdf_fitted_log8)
#plot data histogram and pdf curve
h = plt.hist(maize_kgha_log, bins=50, density=True)
plt.plot(xm1, pdf_fitted8, lw=2, label="Fitted normal distribution on log")
plt.legend()
plt.title('log Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.show()

#one in all plot
h = plt.hist(maize_kgha, bins=50, density=True)
plt.plot(xm, pdf_fitted, lw=2, label="Fitted Lognormal distribution")
plt.plot(xm, pdf_fitted2, lw=2, label="Fitted Exponential distribution")
plt.plot(xm, pdf_fitted3, lw=2, label="Fitted Weibull distribution")
plt.plot(xm, pdf_fitted4, lw=2, label="Fitted Normal distribution")
plt.plot(xm, pdf_fitted5, lw=2, label="Fitted Halfnormal distribution")
plt.plot(xm, pdf_fitted6, lw=2, label="Fitted Gamma distribution")
plt.plot(xm, pdf_fitted7, lw=2, label="Fitted Inverse Gamma distribution")
plt.legend()
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=20000)
plt.show()


#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_listm, pdf_listm, param_dictm)
#       Distribution  loglikelihood           AIC           BIC
#7  normal on log  -7.765280e+05  1.553072e+06  1.553162e+06
#5          Gamma  -5.184958e+06  1.036993e+07  1.037002e+07
#1    exponential  -5.200183e+06  1.040038e+07  1.040047e+07
#6  Inverse Gamma  -5.204101e+06  1.040822e+07  1.040831e+07
#4     halfnormal  -5.204261e+06  1.040854e+07  1.040863e+07
#3         normal  -5.356897e+06  1.071381e+07  1.071390e+07
#0        lognorm  -6.250698e+06  1.250141e+07  1.250150e+07
#2        weibull  -6.429530e+06  1.285908e+07  1.285917e+07
#best fit so far: normal on log values by far, then Gamma on non-log

'''
Load factor data and extract zeros
'''
m_pesticides=pd.read_csv(params.geopandasDataDir + 'CornPesticidesHighRes.csv')
print(m_pesticides.columns)
print(m_pesticides.head())
fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv') #kg/m²
print(fertilizer.columns)
print(fertilizer.head())
fertilizer_man=pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv') #kg/km²
print(fertilizer_man.columns)
print(fertilizer_man.head())
irr_t=pd.read_csv(params.geopandasDataDir + 'FracIrrigationAreaHighRes.csv')
crop = pd.read_csv(params.geopandasDataDir + 'FracCropAreaHighRes.csv')
irr_rel=pd.read_csv(params.geopandasDataDir + 'FracReliantHighRes.csv')
#the data is off because it wasn't divided by 25 (for the upsampling of 0.5 degree to 5 arcmin) so has to be done now
#irr = irrigation.iloc[:,[3,4]]/25
#irr_l0 = irr_lowres.loc[irr_lowres['area'] > 0]
print(irr_t.columns)
print(irr_t.head())
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
print(m_pesticides.loc[1491959])
print(fertilizer.loc[1491959])
print(fertilizer_man.loc[1491959])
print(irr_tot.loc[1491959])
print(tillage.loc[1491959])
print(aez.loc[1491959])
print(maize_yield.loc[1491959])

test_p = m_pesticides.loc[m_pesticides['total_H'].notna()] #555307
test_p = m_pesticides.loc[m_pesticides['total_H'] < 0] #0 as expected

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
datam_raw = {"lat": maize_yield.loc[:,'lats'],
		"lon": maize_yield.loc[:,'lons'],
		"area": maize_yield.loc[:,'growArea'],
        "Y": maize_yield.loc[:,'yield_kgPerHa'],
		"n_fertilizer": fertilizer.loc[:,'n_kgha'],
		"p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_man_prod" : fertilizer_man.loc[:,'produced_kgha'],
        "n_total" : N_total,
        "pesticides_H": m_pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'is_mech'],
        "irrigation_tot": irr_tot.loc[:,'area'],
        "irrigation_rel": irr_rel.loc[:,'tot_reliant'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
dmaize_raw = pd.DataFrame(data=datam_raw)
#select only the rows where the area of the cropland is larger than 0
dm0_raw=dmaize_raw.loc[dmaize_raw['area'] > 0]

#test if the categories are more or less evenly distributed to prevent introducing increased
#artificial error
plt.hist(dm0_raw['thz_class'], bins=50)
#merge the last three classes into one
#class 1 is by far the largest
plt.hist(dm0_raw['mst_class'], bins=50)
#could possibly combine class 1 with class 2 and class 7 with class 6
#maybe don't merge class 1 because there could be a strong interaction with irrigation
plt.hist(dm0_raw['soil_class'], bins=50)
#as discussed below: get rid of urban and water class as those are erronous
#can't merge the first two classes as they represent two different things:
    #class 1: slope limitations, class 2: hydrological limitations

#test if there are cells with 0s for the AEZ classes (there shouldn't be any)
m_testt = dm0_raw.loc[dm0_raw['thz_class'] == 0] #only 42 0s
m_testm = dm0_raw.loc[dm0_raw['mst_class'] == 0] #only 42 0s
m_tests = dm0_raw.loc[dm0_raw['soil_class'] == 0]
#2583 0s probably due to the original soil dataset being in 30 arcsec resolution:
    #land/ocean boundaries, especially of islands, don't always align perfectly

#test if certain classes of the AEZ aren't present in the dataset because they
#represent conditions which aren't beneficial for plant growth
#thz_class: test Arctic and Bor_cold_with_permafrost
m_test_t9 = dm0_raw.loc[dm0_raw['thz_class'] == 9]
#19 with Boreal and permafrost: reasonable
m_test_t10 = dm0_raw.loc[dm0_raw['thz_class'] == 10]
#246 with Arctic: is reasonable

#mst_class: test LPG<60days
m_test_m = dm0_raw.loc[dm0_raw['mst_class'] == 1]
#18264 in LPG<60 days class: probably due to irrigation

#soil class: test urban, water bodies and very steep class
m_test_s1 = dm0_raw.loc[dm0_raw['soil_class'] == 1]
#15541 in very steep class: makes sense, there is marginal agriculture in
#agricultural outskirts
m_test_s7 = dm0_raw.loc[dm0_raw['soil_class'] == 7]
#3223 in water class: this doesn't make sense but also due to resolution
#I think these should be substituted
m_test_s8 = dm0_raw.loc[dm0_raw['soil_class'] == 8]
#3496 in urban class: probably due to finer resolution in soil class, e.g. course of 
#the Nile is completely classified with yield estimates even though there are many urban areas
#Question: should the urban datapoints be taken out due to them being unreasonable? But then again
#the other datasets most likely contain values in these spots as well (equally unprecise), so I would
#just lose information
#I could substitute them like the water bodies

#test mech dataset values
m_test_mech0 = dm0_raw.loc[dm0_raw['mechanized'] == 0] #157445, now: 169953
m_test_mech1 = dm0_raw.loc[dm0_raw['mechanized'] == 1] #278059, now: 297566
m_test_mechn = dm0_raw.loc[dm0_raw['mechanized'] == -9] #126276, now: 94261
#this is a problem: -9 is used as NaN value and there are way, way too many
#less NaN's than before, because tillage is combined now for all crops

m_test_f = dm0_raw.loc[dm0_raw['n_fertilizer'] < 0] #8770 NaN's, 30909 0s
m_test_pf = dm0_raw.loc[dm0_raw['p_fertilizer'] < 0] #8770 NaN's, 36104 0s
m_test_man = dm0_raw.loc[dm0_raw['n_manure'] < 0] #21460 0s, but 0 NaN's! 
m_test_p = dm0_raw.loc[dm0_raw['pesticides_H'].isna()] #166565 NaN's
m_test_p0 = dm0_raw.loc[dm0_raw['pesticides_H'] == 0] #0 0s
m_test_it = dm0_raw.loc[dm0_raw['irrigation_tot'] == 0] #213546 0s, but 0 NaN's!
#I was wondering if we could just leave the 0s or if we have to differentiate between 0s
#and NaNs but I came to the conclusion that for the irrigation dataset the number of 0s
#is probably quite accurate
m_test_ir = dm0_raw.loc[dm0_raw['irrigation_rel'].isna()] #69139 0s, 106636 NaN's
#this is odd but the values are as well: are about 50% for almost all cells (that I've seen)

#replace 0s in the moisture, climate and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
dm0_raw['thz_class'] = dm0_raw['thz_class'].replace(0,np.nan)
dm0_raw['mst_class'] = dm0_raw['mst_class'].replace(0,np.nan)
dm0_raw['soil_class'] = dm0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
dm0_raw['thz_class'] = dm0_raw['thz_class'].replace([9,10],8)
print(dm0_raw.loc[1511426])
print(dm0_raw.loc[7190035])
print(dm0_raw.loc[1643058])

#NaN values throw errors in the regression, they need to be handled beforehand
dm0_raw['pesticides_H'] = dm0_raw['pesticides_H'].replace(np.nan, -9)
dm0_raw['irrigation_rel'] = dm0_raw['irriagtion_rel'].replace(np.nan, -9)
#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
#this is fine for now as there most likely won't be any NaN values at full resolution
dm0_raw = dm0_raw.fillna(method='ffill')

#how do I handle NaN's in fertilizer and in mechanized? I could also use the ffill method
#but that would be quite inaccurate, KNN might be harder to implement
#other option is to delete the respective rows
#apparently there are only 0s around the NaN values in the fertilizer data anyway
#so I could just fill them up with 0s
#nvm there are only 0s in the dataset...
#n_manure hist and scatter look a lot better on log scale
#it's hard to determine if there is a relationship between the climate classes and yield

#select only the rows, where the column mechanized is non-NaN
dm0_test = dm0_raw.loc[dm0_raw['mechanized'] > -9]
test_f = dm0_test.loc[dm0_test['n_fertilizer'] == 0] #19064
test_pf = dm0_test.loc[dm0_test['p_fertilizer'] == 0] #22847
test_man = dm0_test.loc[dm0_test['n_manure'] == 0] #14869
test_p = dm0_test.loc[dm0_test['pesticides_H'] == 0] #107107
dm0_test1 = dm0_test.loc[dm0_test['pesticides_H'] < 0]
test1_f = dm0_test1.loc[dm0_test1['n_fertilizer'] == 0] #5965

#fill in the remaining couple of nans at the top of mechanized column
dm0_raw['mechanized'] = dm0_raw['mechanized'].fillna(1)

#Try two different methods to handle the data:
#1: eliminate the respective rows:
dm0_elim = dm0_raw.loc[dm0_raw['mechanized'] > -9]
dm0_elim = dm0_elim.loc[dm0_elim['n_fertilizer'] > -9]
dm0_elim = dm0_elim.loc[dm0_elim['pesticides_H'] > -9]
m_test_p1 = dm0_elim.loc[dm0_elim['pesticides_H'] < 0]
#m_test_f1 = dm0_elim.loc[dm0_elim['pesticides_H'] < 0] #8770 NaN's, 30909 0s


#2: use ffill for the NaN values
dm0_raw['mechanized'] = dm0_raw['mechanized'].replace(-9,np.nan)
dm0_raw['pesticides_H'] = dm0_raw['pesticides_H'].replace(-9,np.nan)
dm0_raw.loc[dm0_raw['n_fertilizer']<0] = np.nan
dm0_raw.loc[dm0_raw['p_fertilizer']<0] = np.nan
dm0_raw.loc[dm0_raw['n_total']<0] = np.nan
dm0_raw = dm0_raw.fillna(method='ffill')
dm0_raw['mechanized'] = dm0_raw['mechanized'].fillna(1)

#for logging, replace 0s in n_man with a value a magnitude smaller than the smallest
#real value
min_dm0_log=dm0_raw[dm0_raw.iloc[:,0:16]>0].min()
dm0_flog = dm0_raw
dm0_flog['n_manure'] = dm0_flog['n_manure'].replace(0,0.0000000003)
dm0_flog['n_fertilizer'] = dm0_flog['n_fertilizer'].replace(0,0.1)
dm0_flog['p_fertilizer'] = dm0_flog['p_fertilizer'].replace(0,0.09)
dm0_flog['n_total'] = dm0_flog['n_total'].replace(0,0.0000000009)
dm0_elog = dm0_elim
dm0_elog['n_manure'] = dm0_elog['n_manure'].replace(0,0.00000000001)
dm0_elog['n_fertilizer'] = dm0_elog['n_fertilizer'].replace(0,0.1)
dm0_elog['p_fertilizer'] = dm0_elog['p_fertilizer'].replace(0,0.09)
dm0_elog['n_total'] = dm0_elog['n_total'].replace(0,0.0000000009)

###############Outliers###########################
m_out_f = dm0_elim.loc[dm0_elim['n_fertilizer'] > 400] #only 78 left
m_out_p = dm0_elim.loc[dm0_elim['p_fertilizer'] > 100] #169
m_out_man = dm0_elim.loc[dm0_elim['n_manure'] > 250] #35; 69 bei 200
m_out_prod = dm0_elim.loc[dm0_elim['n_man_prod'] > 1000] #32
m_out_n = dm0_elim.loc[(dm0_elim['n_manure'] > 250) | (dm0_elim['n_fertilizer'] > 400)] #has to be 78+35-1=112

m_mman = dm0_elim['n_manure'].mean() #5.103560635784215
m_medman = dm0_elim['n_manure'].median() #2.6500869750976563

dm0_qt = dm0_elim.quantile([.1, .5, .75, .9, .95, .99, .999])

#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('dm0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=dm0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=dm0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=dm0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=dm0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=dm0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=dm0_elim, x='Y')

ax = sb.boxplot(x=dm0_elim["Y"], orient='v')
ax = sb.boxplot(x=dm0_elim["n_fertilizer"])
ax = sb.boxplot(x=dm0_elim["p_fertilizer"])
ax = sb.boxplot(x=dm0_elim["n_manure"])
ax = sb.boxplot(x=dm0_elim["n_total"])
ax = sb.boxplot(x=dm0_elim["pesticides_H"])
ax = sb.boxplot(x=dm0_elim["irrigation_tot"])
ax = sb.boxplot(x=dm0_elim["irrigation_rel"])
ax = sb.boxplot(x="mechanized", y='Y', data=dm0_elim)
ax = sb.boxplot(x="thz_class", y='Y', hue='mechanized', data=dm0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=dm0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=dm0_elim)

#replace nonsense values in fertilizer and manure datasets
dm0_elim.loc[dm0_elim['n_fertilizer'] > 400, 'n_fertilizer'] = np.nan
dm0_elim.loc[dm0_elim['p_fertilizer'] > 100, 'p_fertilizer'] = np.nan
dm0_elim.loc[dm0_elim['n_manure'] > 250, 'n_manure'] = np.nan
#dm0_elim.loc[dm0_elim['n_man_prod'] > 1000, 'n_man_prod'] = np.nan
dm0_elim = dm0_elim.fillna(method='ffill')
dm0_elim['n_total'] = dm0_elim['n_manure'] + dm0_elim['n_fertilizer']



###############################################################################
############Loading log transformed values for all variables##################
##############################################################################


#using log values for the input into the regression
#unfortunately the ln of 0 is not defined
#just keeping the 0 would skew the results as that would imply a 1 in the data when there is a 0
#could just use the smallest value of the dataset as a substitute?
data_log = {"Y": np.log(dm0_elog.loc[:,'Y']),
		"n_fertilizer": np.log(dm0_elog.loc[:,'n_fertilizer']),
		"p_fertilizer": np.log(dm0_elog.loc[:,'p_fertilizer']),
        "n_manure": np.log(dm0_elog.loc[:,'n_manure']),
        "n_total" : np.log(dm0_elog.loc[:,'n_total']),
        "pesticides_H": np.log(dm0_elog.loc[:,'pesticides_H']),
        "mechanized": dm0_elog.loc[:,'mechanized'],
        "irrigation": np.log(dm0_elog.loc[:,'area']), #does the log belong here, I believe not because it's a fraction
        "thz_class" : dm0_elog.loc[:,'thz_class'],
        "mst_class" : dm0_elog.loc[:,'mst_class'],
        "soil_class": dm0_elog.loc[:,'soil_class']
		}


dm0_log = pd.DataFrame(data=data_log)

'''
#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
#####Get dummies##########
mdum_mst = pd.get_dummies(dm0_raw['mst_class'])
mdum_thz = pd.get_dummies(dm0_raw['thz_class'])
mdum_soil = pd.get_dummies(dm0_raw['soil_class'])
#####Rename Columns##########
mdum_mst = mdum_mst.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
mdum_thz = mdum_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor+Arctic"}, errors="raise")
mdum_soil = mdum_soil.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
####RAW#########
dmaize_d_raw = pd.concat([dm0_raw, mdum_mst, mdum_thz, mdum_soil], axis='columns')
dmaize_d_elim = pd.concat([dm0_elim, mdum_mst, mdum_thz, mdum_soil], axis='columns')
dmaize_d_elim = dmaize_d_elim.dropna()
######LOG#########
dmaize_d_log = pd.concat([dm0_log, mdum_mst, mdum_thz, mdum_soil], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
#####RAW#####
dmaize_dum_raw = dmaize_d_raw.drop(['365+days','Bor+Arctic', 'L1_irr'], axis='columns')
dmaize_dum_elim = dmaize_d_elim.drop(['365+days','Bor+Arctic', 'L1_irr'], axis='columns')
########LOG#######
dmaize_dum_log = dmaize_d_log.drop(['365+days', 'Bor+Arctic', 'L1_irr'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dmaize_val_raw = dmaize_dum_raw.sample(frac=0.2, random_state=2705) #RAW
dmaize_val_elim = dmaize_dum_elim.sample(frac=0.2, random_state=2705) #RAW
dmaize_val_log = dmaize_dum_log.sample(frac=0.2, random_state=2705) #LOG
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dmaize_fit_raw = dmaize_dum_raw.drop(dmaize_val_raw.index) #RAW
dmaize_fit_elim = dmaize_dum_elim.drop(dmaize_val_elim.index) #RAW
dmaize_fit_log = dmaize_dum_log.drop(dmaize_val_log.index) #LOG
'''
dmaize_val_raw = dm0_raw.sample(frac=0.2, random_state=2705) #RAW
dmaize_val_elim = dm0_elim.sample(frac=0.2, random_state=2705) #RAW
dmaize_val_log = dm0_log.sample(frac=0.2, random_state=2705) #LOG
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dmaize_fit_raw = dm0_raw.drop(dmaize_val_raw.index) #RAW
dmaize_fit_elim = dm0_elim.drop(dmaize_val_elim.index) #RAW
dmaize_fit_log = dm0_log.drop(dmaize_val_log.index) #LOG
'''
#select all rows from dmaize_log for which the column growArea has a value greater than zero
dm0_log=dmaize_log.loc[dmaize_log['area'] > 0]
'''
#the data contains -inf values because the n+p+pests+irrigation columns contain 0 values for which ln(x) is not defined 
#calculate the minimum values for each column disregarding -inf values to see which is the lowest value in the dataset (excluding lat & lon)
min_dm0_log=dm0_log[dm0_log.iloc[:,0:11]>-inf].min()
test_log = dm0_log.loc[dm0_log['n_fertilizer'] == -inf] #10398
#replace the -inf values with the minimum of each respective column + ~1 : this is done to achieve a distinction between very small
#values and actual zeros
dm0_log['p_fertilizer'] = dm0_log['p_fertilizer'].replace(-inf,-3)
dm0_log['n_fertilizer'] = dm0_log['n_fertilizer'].replace(-inf,-3)
dm0_log['n_manure'] = dm0_log['n_manure'].replace(-inf,-22)
dm0_log['n_total'] = dm0_log['n_total'].replace(-inf,-14)
dm0_log['pesticides_H'] = dm0_log['pesticides_H'].replace(-inf,-11)

#check distribution of AEZ factors in the historgrams
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

plt.hist(dm0_log['soil_class'], bins=50)
plt.hist(dm0_log['mst_class'], bins=50)
plt.hist(dm0_log['thz_class'], bins=50)
'''
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
'''
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

dm0_elim.plot.scatter(x = 'n_total', y = 'Y')
dm0_raw.plot.scatter(x = 'p_fertilizer', y = 'yield')
dm0_elim.plot.scatter(x = 'n_total', y = 'Y')
dm0_raw.plot.scatter(x = 'mechanized', y = 'yield')
dm0_raw.plot.scatter(x = 'thz_class', y = 'yield')
dm0_raw.plot.scatter(x = 'soil_class', y = 'yield')

#scatterplots and histograms for #LOG variables
dm0_log.plot.scatter(x = 'n_fertilizer', y = 'yield')
dm0_log.plot.scatter(x = 'p_fertilizer', y = 'yield')
dm0_log.plot.scatter(x = 'pesticides_H', y = 'yield')
dm0_log.plot.scatter(x = 'mechanized', y = 'yield')
dm0_log.plot.scatter(x = 'n_total', y = 'yield')
dm0_elim.plot.scatter(x = 'irrigation_tot', y = 'Y')
dm0_log.plot.scatter(x = 'thz_class', y = 'yield')
dm0_elim.plot.scatter(x = 'mst_class', y = 'Y')
dm0_log.plot.scatter(x = 'soil_class', y = 'yield')

plt.hist(dm0_elim['n_fertilizer'], bins=50)
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

param6 = stats.norm.fit(dm0_elim['Y'])
#param_list.append(param6)
param_dictm["Values"].append(param6)
print(param6)
#calculate pdf
pdf_fitted6 = stats.norm.pdf(xm, *param6)
#calculate log pdf and store it in the list
pdf_fitted_log6 = stats.norm.logpdf(dm0_elim['Y'], *param6)
pdf_listm.append(pdf_fitted_log6)
#plot data histogram and pdf curve
h = plt.hist(dm0_elim['Y'], bins=50, density=True)
plt.plot(xm, pdf_fitted6, lw=2, label="Fitted Gamma distribution")
plt.legend()
plt.show()


#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
#####Get dummies##########
mdum_mst = pd.get_dummies(dm0_test1['mst_class'])
mdum_thz = pd.get_dummies(dm0_test1['thz_class'])
mdum_soil = pd.get_dummies(dm0_test1['soil_class'])
#####Rename Columns##########
mdum_mst = mdum_mst.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
mdum_thz = mdum_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor+Arctic"}, errors="raise")
mdum_soil = mdum_soil.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
####RAW#########
dmaize_d_raw = pd.concat([dm0_test1, mdum_mst, mdum_thz, mdum_soil], axis='columns')
######LOG#########
dmaize_d_log = pd.concat([dm0_log, mdum_mst, mdum_thz, mdum_soil], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
#####RAW#####
dmaize_dum_raw = dmaize_d_raw.drop(['365+days', 
                      'Bor+Arctic', 'L1_irr'], axis='columns')
########LOG#######
dmaize_dum_log = dmaize_d_log.drop(['365+days', 
                      'Bor+Arctic', 'L1_irr'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dmaize_val_raw = dmaize_dum_raw.sample(frac=0.2, random_state=2705) #RAW
dmaize_val_log = dmaize_dum_log.sample(frac=0.2, random_state=2705) #LOG
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dmaize_fit_raw = dmaize_dum_raw.drop(dmaize_val_raw.index) #RAW
dmaize_fit_log = dmaize_dum_log.drop(dmaize_val_log.index) #LOG

##################Collinearity################################

###########RAW#################

grid = sb.PairGrid(data= dmaize_fit_raw,
                    vars = ['n_fertilizer', 'p_fertilizer', 'n_total',
                    'pesticides_H', 'mechanized', 'irrigation'], height = 4)
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', 
                     edgecolor = 'k')
grid = grid.map_lower(sb.kdeplot, cmap = 'Reds')
#wanted to display the correlation coefficient in the lower triangle but don't know how
#grid = grid.map_lower(corr)

sb.pairplot(dmaize_dum_raw)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dmaize_cor_raw = dmaize_fit_raw.drop(['lat', 'lon', 'area', 'yield'], axis='columns')
#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(dmaize_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp = dmaize_cor_raw.corr(method='spearman')
print(sp)
sp.iloc[0,1:5]
sp.iloc[1,2:5]
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

X = add_constant(dmaize_cor_elim)
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


X = add_constant(dmaize_cor_elim)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
'''
Out[195]: 
const                706.047259
p_fertilizer           6.075385
n_total                6.055918
pesticides_H           2.094650
mechanized             1.441289
irrigation_tot         1.896832
LGP<60days             2.803385
60-120days             8.540601
120-180days           20.675855
180-225days           19.111780
225-270days           17.741380
270-365days           23.367627
Trop_low             133.287284
Trop_high             26.817987
Sub-trop_warm         39.693882
Sub-trop_mod_cool     57.442973
Sub-trop_cool         35.391055
Temp_mod              68.672860
Temp_cool             96.951993
S1_very_steep          1.388502
S2_hydro_soil          1.319893
S3_no-slight_lim       3.887261
S4_moderate_lim        3.940900
S5_severe_lim          1.785457
dtype: float64
'''

#drop aggregated climate classes
cor_elim = dmaize_cor_elim.drop(['270-365days', 'Trop_low'], axis='columns')
cor_elim = dmaize_cor_elim.drop(['LGP<60days', 'Trop_high'], axis='columns')
cor_elim = dmaize_cor_elim.drop(['LGP<60days', '60-120days', '120-180days',
                                    '180-225days', '225-270days', 
                                    '270-365days'], axis='columns')
X1 = add_constant(cor_elim)
pd.Series([variance_inflation_factor(X1.values, i) 
               for i in range(X1.shape[1])], 
              index=X1.columns)
#thz and mst factor levels are pretty highly correlated


######################TEST#########################

test_M = dm0_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel'], axis='columns')
test_M['thz_class'] = test_M['thz_class'].replace([8],7)

test_M['mst_class'] = test_M['mst_class'].replace([2],1)
test_M['mst_class'] = test_M['mst_class'].replace([7],6)

plt.hist(dm0_elim['soil_class'])
bor_test = dm0_elim.loc[dm0_elim['thz_class'] == 8] #602

md_mst = pd.get_dummies(test_M['mst_class'])
md_thz = pd.get_dummies(test_M['thz_class'])

md_mst = md_mst.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
md_thz = md_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
test_M = pd.concat([test_M, md_mst, md_thz, mdum_soil], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
test_M.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns', inplace=True)

test_cor_elim = test_M.drop(['thz_class','mst_class', 'soil_class'], axis='columns')

#drop dummy variables
cor_test = test_cor_elim.loc[:,['n_manure', 'mechanized', 'thz_class', 'mst_class', 
                                   'soil_class']]
X2 = add_constant(test_cor_elim)
pd.Series([variance_inflation_factor(X2.values, i) 
               for i in range(X2.shape[1])], 
              index=X2.columns)

plt.hist(test_M['mst_class'], bins=50)
ax = sb.boxplot(x=test_M["mst_class"], y=dm0_elim['Y'])
plt.ylim(0,20000)

'''
mst_test

const                606.320177
p_fertilizer           6.070693
n_total                6.032359
pesticides_H           2.087932
mechanized             1.433503
irrigation_tot         1.884955
LGP<120days            1.469800
120-180days            1.696699
180-225days            1.631926
225-270days            1.439098
Trop_low             134.007259
Trop_high             27.017447
Sub-trop_warm         39.723240
Sub-trop_mod_cool     57.521198
Sub-trop_cool         35.687529
Temp_mod              69.075323
Temp_cool             97.349586
S1_very_steep          1.386676
S2_hydro_soil          1.321069
S3_no-slight_lim       3.883745
S4_moderate_lim        3.946254
S5_severe_lim          1.785712
dtype: float64

thz_test
 
const                35.974383
p_fertilizer          6.066513
n_total               6.024629
pesticides_H          2.085353
mechanized            1.432192
irrigation_tot        1.882304
LGP<120days           1.469619
120-180days           1.696031
180-225days           1.631311
225-270days           1.438670
Trop_low              2.907152
Trop_high             1.397176
Sub-trop_warm         1.605076
Sub-trop_mod_cool     1.620597
Sub-trop_cool         1.429233
Temp_mod              1.575785
S1_very_steep         1.384446
S2_hydro_soil         1.320858
S3_no-slight_lim      3.883476
S4_moderate_lim       3.943208
S5_severe_lim         1.783290
dtype: float64
'''



######################Regression##############################

'''
#rename Y variable
dmaize_fit_raw = dmaize_fit_raw.rename(columns={'yield':'Y'}, errors='raise')
dmaize_fit_elim = dmaize_fit_elim.rename(columns={'yield':'Y'}, errors='raise')
'''

#determine models
#Normal distribution
mod_rawn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dmaize_fit_raw)
mod_elimn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dmaize_fit_elim)
#Gamma distribution
mod_rawg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_raw, 
              family=sm.families.Gamma(link=sm.families.links.log))
mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
mod_raw0 = smf.glm(formula='Y ~ 1', data=dmaize_fit_raw, family=sm.families.Gamma(link=sm.families.links.log))
mod_elim0 = smf.glm(formula='Y ~ 1', data=dmaize_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
fit_rawn = mod_rawn.fit()
fit_elimn = mod_elimn.fit()
fit_rawg = mod_rawg.fit()
fit_elimg = mod_elimg.fit()
fit_raw0 = mod_raw0.fit()
fit_elim0 = mod_elim0.fit()
#print results
print(fit_rawn.summary())
print(fit_elimn.summary())
print(fit_rawg.summary())
print(fit_elimg.summary())
print(fit_raw0.summary())
print(fit_elim0.summary())

#calculate pseudo R
pseudoR_raw = 1-(234160/358480)  #0.24823 0.0952 0.347
pseudoR_elim = 1-(125780/199010) #0.29316 0.0982 0.348 0.366
print(pseudoR_raw)
print(pseudoR_elim)

val_raw = dmaize_val_raw.drop(['lat', 'lon', 'area', 'Y', 'n_manure', 'n_fertilizer'], axis='columns')
val_elim = dmaize_val_elim.drop(['lat', 'lon', 'area', 'Y', 'n_manure', 'n_fertilizer'], axis='columns')

pred_raw = fit_rawn.predict(val_raw)
pred_elim = fit_elimn.predict(val_elim)
#glm: like this they aren't valid -> link function
pred_rawg = fit_rawg.predict(val_raw)
pred_elimg = fit_elimg.predict(val_elim)

r2_score(dmaize_val_raw['Y'], pred_raw) #0.3297567051138425
r2_score(dmaize_val_elim['Y'], pred_elim) #0.34027102028441003
r2_score(dmaize_val_raw['Y'], pred_rawg) 
r2_score(dmaize_val_elim['Y'], pred_elimg) 

plt.scatter(pred_raw, dmaize_val_raw['Y'])
plt.scatter(pred_rawg, dmaize_val_raw['Y'])
plt.scatter(pred_elim, dmaize_val_elim['Y'])
plt.scatter(pred_elimg, dmaize_val_elim['Y'])

plt.scatter(dmaize_val_raw['n_total'], pred_raw)
plt.scatter(dmaize_val_raw['n_total'], dmaize_val_raw['Y'])
plt.scatter(dmaize_val_elim['n_total'], pred_elim)
plt.scatter(dmaize_val_elim['n_total'], dmaize_val_elim['Y'])
plt.scatter(dmaize_val_raw['n_total'], pred_rawg)
plt.scatter(dmaize_val_elim['n_total'], pred_elimg)

an_raw = pd.concat([dmaize_val_raw, pred_raw, pred_rawg], axis='columns')
an_raw = an_raw.rename(columns={0:"pred_raw", 1:"pred_rawg"}, errors="raise")
sb.lmplot(x='pred_raw', y='Y', data=an_raw)

plt.hist(pred_elimg, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

##############LOG################################
#rename Y variable
dmaize_fit_log= dmaize_fit_log.rename(columns={'yield':'Y'}, errors='raise')
#determine models
mod_logn = smf.ols(formula=' Y ~ n_total + p_fertilizer + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dmaize_fit_log)           
#Fit models
fit_logn = mod_logn.fit()
#print results
print(fit_logn.summary())







#R-style formula
#doesn't work for some reason... I always get parsing errors and I don't know why
mod = smf.ols(formula=' yield ~ n_total + pesticides_H + mechanized + irrigation', data=dmaize_fit_raw)

mod = smf.ols(formula='yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dmaize_fit_raw)

dmaize_fit_raw = dmaize_fit_raw.rename(columns={'yield':'Y'}, errors='raise')
dmaize_fit_log= dmaize_fit_log.rename(columns={'yield':'Y'}, errors='raise')
modas = smf.glm(formula='Y ~ n_total + pesticides_H + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_raw, 
              family=sm.families.Gamma())
modas1 = smf.ols(formula='Y ~ n_total + pesticides_H + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_raw 
              )
modas1_fit = modas1.fit()
print(modas1.fit().summary())

mod24 = smf.glm(formula='Y ~ 1', data=dmaize_fit_raw, family=sm.families.Gamma())
m24 = mod24.fit()
print(m24.summary())

pseudoR = 1-(148190/189950)    

#use patsy to create endog and exog matrices in an Rlike style
y, X = dmatrices('yield ~ n_fertilizer + pesticides_H + mechanized + irrigation', data=dmaize_fit_raw, return_type='dataframe')


#Python design matrices
#define x and y dataframes
#Y containing only yield
m_endog_raw = dmaize_fit_raw.iloc[:,3] #RAW
m_endog_elim = dmaize_fit_elim.iloc[:,3] #RAW
m_endog_log = dmaize_fit_log.iloc[:,0] #LOG
#X containing all variables
m_exog_alln_raw = dmaize_fit_raw.drop(['Y', 'lat', 'lon', 'area', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #RAW
m_exog_alln_elim = dmaize_fit_elim.drop(['Y', 'lat', 'lon', 'area', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #RAW
m_exog_alln_log = dmaize_fit_log.drop(['Y', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #LOG


####testing regression
#determining the models
###RAW###
mod_alln_raw = sm.OLS(m_endog_raw, m_exog_alln_raw)
mod_alln_elim = sm.OLS(m_endog_elim, m_exog_alln_elim)
mod_alln_rawg = sm.GLM(m_endog_raw, m_exog_alln_raw, family=sm.families.Gamma())
mod_alln_elimg = sm.GLM(m_endog_elim, m_exog_alln_elim, family=sm.families.Gamma())
###LOG
mod_alln_log = sm.OLS(m_endog_log, m_exog_alln_log)
####LOG DEPENDENT####
mod_alln_mix = sm.OLS(m_endog_log, m_exog_alln_raw)

#fitting the models
#####RAW#####
mod_res_alln_raw = mod_alln_raw.fit(method='qr')
mod_res_alln_elim = mod_alln_elim.fit(method='qr')
mod_res_alln_rawg = mod_alln_rawg.fit()
mod_res_alln_elimg = mod_alln_elimg.fit()
####LOG####
mod_res_alln_log = mod_alln_log.fit(method='qr')
####LOG DEPENDENT####
mod_res_alln_mix = mod_alln_mix.fit()

#printing the results
print(mod_res_alln_raw.summary())
print(mod_res_alln_elim.summary())
print(mod_res_alln_rawg.summary())
print(mod_res_alln_elimg.summary())

print(mod_res_alln_log.summary())

print(mod_res_alln_mix.summary())

#define x and y dataframes
#Y containing only yield
m_endog_raw = dmaize_fit_raw.iloc[:,3] #RAW
m_endog_log = dmaize_fit_log.iloc[:,0] #LOG
#X containing all variables
m_exog_alln_raw = dmaize_fit_raw.drop(['Y', 'lat', 'lon', 'area', 'n_total', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #RAW
m_exog_alln_log = dmaize_fit_log.drop(['Y', 'n_total', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #LOG
#test with n total and p
m_exog_np_raw = dmaize_fit_raw.drop(['Y', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #RAW
m_exog_np_log = dmaize_fit_log.drop(['Y', 'n_fertilizer', 'n_manure', 'mst_class', 'thz_class', 'soil_class'], axis='columns')  #LOG
#test with n total without p
m_exog_n_log = dmaize_fit_raw.drop(['Y', 'lat', 'lon', 'area', 'n_fertilizer', 'n_manure', 'p_fertilizer', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #RAW
m_exog_n_raw = dmaize_fit_log.drop(['Y', 'n_fertilizer', 'n_manure', 'p_fertilizer', 'mst_class', 'thz_class', 'soil_class'], axis='columns') #LOG
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


print(mod_res_alln_log.summary())

print(mod_res_alln_mix.summary())
print(mod_res_np_mix.summary())
print(mod_res_n_mix.summary())


##########RESIDUALS#############

sm.graphics.plot_regress_exog(modas1_fit, 'C(thz_class)[T.2.0]')
plt.show()

modas1_fitted = modas1_fit.fittedvalues
mod_res_alln_raw_fitted = mod_res_alln_raw.fittedvalues
mod_res_alln_log_fitted = mod_res_alln_log.fittedvalues

modas1_resd = modas1_fit.resid
mod_res_alln_raw_resd = mod_res_alln_raw.resid
mod_res_alln_log_resd = mod_res_alln_log.resid

modas_abs_resid = np.abs(modas1_resd)

plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sb.residplot(modas1_fitted, 'Y', data=dmaize_fit_raw, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = modas_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i, 
                               xy=(modas1_fitted[i], 
                                   modas1_resd[i]))


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


#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

#take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
#and available manure correctly
LoI_melim = dm0_raw

LoI_melim['mechanized'] = LoI_melim['mechanized'].replace(-9,np.nan)
LoI_melim['pesticides_H'] = LoI_melim['pesticides_H'].replace(-9,np.nan)

############ Mechanised ##########################

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_melim['mechanized_y2'] = LoI_melim['mechanized'].replace(1,0)

############ N fertilizer #########################

mn_drop= LoI_melim[((LoI_melim['mechanized'].isna())|(LoI_melim['pesticides_H'].isna()))
                & (LoI_melim['n_fertilizer']<0)].index
LoI_melim_pn = LoI_melim.drop(mn_drop)

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_melim_pn.loc[LoI_melim_pn['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
LoI_melim_pn.loc[LoI_melim_pn['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
LoI_melim_pn[['n_fertilizer','p_fertilizer']] = LoI_melim_pn[['n_fertilizer','p_fertilizer']].fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
LoI_melim_pn.loc[LoI_melim_pn['n_total'] < 0, 'n_total'] = LoI_melim_pn['n_fertilizer'] + LoI_melim_pn['n_manure']

#drop the nonsense values (99.9th percentile) in the n and p fertilizer columns
LoI_melim_pn = LoI_melim_pn.loc[LoI_melim_pn['n_fertilizer'] < dm0_qt.iloc[12,4]]#~180
LoI_melim_pn = LoI_melim_pn.loc[LoI_melim_pn['p_fertilizer'] < dm0_qt.iloc[12,5]] #~34

#in year 1, there will probably be a slight surplus of N (production>application)
#calculate kg N applied per cell
LoI_melim_pn['n_kg'] = LoI_melim_pn['n_fertilizer']*LoI_melim_pn['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_pn['n_ffrac'] = LoI_melim_pn['n_kg']/(LoI_melim_pn['n_kg'].sum())

#calculate the fraction of total N applied to maize fields on the total N applied
#divide total of maize N by 1000000 to get from kg to thousand t
m_nfert_frac = (LoI_melim_pn['n_kg'].sum())/1000000/118763
#calculate the new total for N maize in year one based on the N total surplus
m_ntot_new = m_nfert_frac * 14477 * 1000000

#calculate the new value of N application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melim_pn['n_fert_y1'] = (m_ntot_new * LoI_melim_pn['n_ffrac']) / LoI_melim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melim_pn['n_fert_y2'] = 0
#LoI_melim_pn.loc[LoI_melim_pn['n_fert_y2'] > 0, 'n_fert_y1'] = 0

############## P Fertilizer #####################

#in year 1, there will probably be a slight surplus of P (production>application)
#calculate kg p applied per cell
LoI_melim_pn['p_kg'] = LoI_melim_pn['p_fertilizer']*LoI_melim_pn['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_pn['p_ffrac'] = LoI_melim_pn['p_kg']/(LoI_melim_pn['p_kg'].sum())

#calculate the fraction of total P applied to maize fields on the total P applied to cropland
#divide total of maize P by 1000000 to get from kg to thousand t
m_pfert_frac = (LoI_melim_pn['p_kg'].sum())/1000000/45858
#calculate the new total for P maize in year one based on the P total surplus
m_ptot_new = m_pfert_frac * 4142 * 1000000

#calculate the new value of P application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melim_pn['p_fert_y1'] = (m_ptot_new * LoI_melim_pn['p_ffrac']) / LoI_melim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melim_pn['p_fert_y2'] = 0

############# N Manure ###################

#drop the rows containing nonsense values (99th percentile) in the manure column
LoI_melim_man = LoI_melim.loc[LoI_melim['n_manure'] < dm0_qt.iloc[10,6]] #~11

#calculate kg N applied per cell: 1,018,425,976.75 kg total
LoI_melim_man['man_kg'] = LoI_melim_man['n_manure']*LoI_melim_man['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_man['n_mfrac'] = LoI_melim_man['man_kg']/(LoI_melim_man['man_kg'].sum())

#calculate the fraction of total N applied to maize fields of the total N applied to cropland
#divide total of maize N by 1000000 to get from kg to thousand t
m_nman_frac = (LoI_melim_man['man_kg'].sum())/1000000/24000

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_melim_man['labor'] = LoI_melim_man['area']/5 #current value (7.4) is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer
#I chose 5 now just because I don't believe 7.4 is correct

#calculate mean excretion rate of each cow in one year: cattle supplied ~ 43.7% of 131000 thousand t
#manure production in 2014, there were ~ 1.008.570.000(Statista)/1.439.413.930(FAOSTAT) 
#heads of cattle in 2014
cow_excr = 131000000000*0.437/1439413930

#calculate available manure based on cow labor demand: 1,278,868,812.065 kg
m_man_av = cow_excr * LoI_melim_man['labor'].sum()
#more manure avialable then currently applied, but that is good as N from mineral
#fertilizer will be missing

#calculate the new value of man application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melim_man['man_fert'] = (m_man_av * LoI_melim_man['n_mfrac']) / LoI_melim_man['area']


########### N total ######################

LoI_melim['N_toty1'] = LoI_melim_pn['n_fert_y1'] + LoI_melim_man['man_fert']
#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_melim['area_LoI'] = LoI_melim['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations

############## Pesticides #####################

LoI_melimp = LoI_melim.loc[LoI_melim['pesticides_H'].notna()]
LoI_melimp = LoI_melimp.loc[LoI_melimp['pesticides_H'] < dm0_qt.iloc[12,9]]#~11
#in year 1, there will probably be a slight surplus of Pesticides (production>application)
#calculate kg p applied per cell
LoI_melimp['pest_kg'] = LoI_melimp['pesticides_H']*LoI_melimp['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melimp['pest_frac'] = LoI_melimp['pest_kg']/(LoI_melimp['pest_kg'].sum())

#calculate the fraction of total pesticides applied to maize fields on the total pesticides applied to cropland
#divide total of maize pesticides by 1000 to get from kg to t
m_pest_frac = (LoI_melimp['pest_kg'].sum())/1000/4190985

#due to missing reasonable data on the pesticide surplus, it is assumed that the
#surplus is in the same range as for P and N fertilizer
frac_pest = ((14477/118763) + (4142/45858))/2
#calculate the new total for pesticides maize in year one based on the pesticides total surplus
m_pestot_new = m_pest_frac * (4190985 * frac_pest) * 1000

#calculate the new value of pesticides application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melimp['pest_y1'] = (m_pestot_new * LoI_melimp['pest_frac']) / LoI_melimp['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melimp['pest_y2'] = 0


########## Irrigation ####################

#in LoI it is assumed that only irrigation which is not reliant on electricity
#can still be maintained
#calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
#(fraction of irrigated cropland reliant on electricity)
LoI_melim['irr_LoI'] = LoI_melim['irrigation_tot'] * (1- LoI_melim['irrigation_rel'])

###########Combine the different dataframes and drop rows with missing values#########

LoI_melim = pd.concat([LoI_melim, LoI_melim_pn['n_fert_y1'], LoI_melim_pn['n_fert_y2'],
                       LoI_melim_pn['p_fert_y1'], LoI_melim_pn['p_fert_y2'],
                       LoI_melim_man['man_fert'], LoI_melimp['pest_y1'], 
                       LoI_melimp['pest_y2']], axis='columns')

#Handle the data by eliminating the rows without data:
LoI_melim = LoI_melim.dropna()

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
#dm0_qt = dm0_elim.quantile([.1, .25, .5, .75, .8,.85, .87, .9, .95,.975, .99,.995, .999,.9999])
#dm0_qt.reset_index(inplace=True, drop=True)
LoI_melim = LoI_melim.loc[LoI_melim['Y'] < dm0_qt.iloc[12,3]] #~12500
#dm0_elim = dm0_elim.loc[dm0_elim['n_man_prod'] < dm0_qt.iloc[12,7]] #~44
LoI_melim = LoI_melim.loc[LoI_melim['n_total'] < dm0_qt.iloc[12,8]] #~195


#########################Prediction of LoI yields#########################

################## Year 1 ##################

#select the rows from LoI_melim which contain the independent variables for year 1
LoI_m_year1 = LoI_melim.iloc[:,[10,13,14,15,17,19,22,25]]
#reorder the columns according to the order in dm0_elim
LoI_m_year1 = LoI_m_year1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_m_year1 = LoI_m_year1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                      'pest_y1':"pesticides_H",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")
#predict the yield for year 1 using the gamma GLM
m_yield_y1 = m_fit_elimg.predict(LoI_m_year1)
#calculate the change rate from actual yield to the predicted yield
m_y1_change = ((m_yield_y1-maize_kgha)/maize_kgha).dropna()

#calculate statistics for yield and change rate

#yield
mmean_y1_weigh = round(np.average(m_yield_y1, weights=LoI_melim['area']),2) #3832.02kg/ha
mmax_y1 = m_yield_y1.max() #10002.44 kg/ha
mmin_y1 = m_yield_y1.min() #691.74 kg/ha

#change rate
mmean_y1c_weigh = round(np.average(m_y1_change, weights=LoI_melim['area']),2) #+0.26 (~+26%)
mmax_y1c = m_y1_change.max() # +105.997 (~+10600%)
mmin_y1c = m_y1_change.min() #-0.94897 (~-95%)

################## Year 2 ##################

#select the rows from LoI_melim which contain the independent variables for year 2
LoI_m_year2 = LoI_melim.iloc[:,[13,14,15,16,19,23,24,26]]
#reorder the columns according to the order in dm0_elim
LoI_m_year2 = LoI_m_year2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_m_year2 = LoI_m_year2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                      'pest_y2':"pesticides_H",'mechanized_y2':"mechanized",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")
#predict the yield for year 2 using the gamma GLM
m_yield_y2 = m_fit_elimg.predict(LoI_m_year2)
#calculate the change from actual yield to the predicted yield
m_y2_change = ((m_yield_y2-maize_kgha)/maize_kgha).dropna()
s = m_y2_change.loc[m_y2_change>0]

#calculate statistics for yield and change rate

#yield
mmean_y2_weigh = round(np.average(m_yield_y2, weights=LoI_melim['area']),2) #2792.08kg/ha
mmax_y2 = m_yield_y2.max() #6551.74kg/ha
mmin_y2 = m_yield_y2.min() #689.79kg/ha

#change rate
mmean_y2c_weigh = round(np.average(m_y2_change, weights=LoI_melim['Y']),2) #+0.02 (~+2%) -0.31?
mmax_y2c = m_y2_change.max() #70.087 (~+7000%)
mmin_y2c = m_y2_change.min() #-0.9503 (~-95%)
m_y2_change.median()
#combine both yields and change rates with the latitude and longitude values
LoI_maize = pd.concat([maize_yield['lats'], maize_yield['lons'], maize_yield['growArea'], m_yield_y1,
                       m_y1_change, m_yield_y2, m_y2_change], axis='columns')
LoI_maize = LoI_maize.rename(columns={0:"m_yield_y1", 1:"m_y1_change", 
                                      2:"m_yield_y2",3:"m_y2_change"}, errors="raise")

round(m_y1_change.quantile([.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99]),2)
round(m_y2_change.quantile([.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99]),2)


#Year 1 yield
3832.02/5525.85 #~69.3% of current average yield
(m_yield_y1 * LoI_melim['area']).sum()
625446340077.8394/825496849007.0315 #~75.8% of current total yield
#Year 2 yield
2792.08/5525.85 #50.5% of current average yield
(m_yield_y2 * LoI_melim['area']).sum()
467475696977.07184/825496849007.0315 #~56.6% of current total yield


ch = LoI_maize.loc[LoI_maize[3]>0]
chm = LoI_maize.loc[LoI_maize[3]<=0]
ch['growArea'].sum()/LoI_maize['growArea'].sum()
38788651.6

#save the dataframe in a csv
LoI_m_year2.to_csv(params.geopandasDataDir + "Test.csv")
LoI_maize.to_csv(params.geopandasDataDir + "LoIMaizeYieldHighRes.csv")

plt.scatter('n_total', 'Y', data=dm0_elim, alpha=0.3)

utilities.create5minASCIIneg(LoI_maize,'m_y1_change',params.asciiDir+'LoIMaizeYieldChange_y1')
utilities.create5minASCIIneg(LoI_maize,'m_yield_y1',params.asciiDir+'LoIMaizeYield_y1')
utilities.create5minASCIIneg(LoI_maize,'m_y2_change',params.asciiDir+'LoIMaizeYieldChange_y2')
utilities.create5minASCIIneg(LoI_maize,'m_yield_y2',params.asciiDir+'LoIMaizeYield_y2')


