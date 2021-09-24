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

#select all rows from maize_yield for which the column growArea has a value greater than zero
maize_nozero=maize_yield.loc[maize_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
maize_kgha=maize_nozero['yield_kgPerHa']

maize_kgha_log=np.log(maize_kgha)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot maize yield distribution in a histogram
plt.hist(maize_kgha, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')


#plot log transformed values of yield_kgPerHa
plt.hist(maize_kgha_log, bins=50)

'''
Fitting of distributions to the data and comparing the fit
'''

#calculate loglik, AIC & BIC for each distribution
#st = stat_ut.stat_overview(dist_listm, pdf_listm, param_dictm)
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
fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv') #kg/m²
fertilizer_man=pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv') #kg/km²
m_tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighResAllCrops.csv')
aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')

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

#################################################################################
##############Loading variables without log to test the effect###################
#################################################################################
data_raw = {"lat": maize_yield.loc[:,'lats'],
		"lon": maize_yield.loc[:,'lons'],
		"area": maize_yield.loc[:,'growArea'],
        "yield": maize_yield.loc[:,'yield_kgPerHa'],
#		"n_fertilizer": fertilizer.loc[:,'n_kgha'],
#		"p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
#        "n_total" : N_total,
#        "pesticides_H": m_pesticides.loc[:,'total_H'],
        "mechanized": m_tillage.loc[:,'is_mech'],
#        "irrigation": irrigation.loc[:,'area'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
dmaize_raw = pd.DataFrame(data=data_raw)
#select only the rows where the area of the cropland is larger than 0
dm0_raw=dmaize_raw.loc[dmaize_raw['area'] > 0]

#replace 0s in the moisture, climate and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
dm0_raw['thz_class'] = dm0_raw['thz_class'].replace(0,np.nan)
dm0_raw['mst_class'] = dm0_raw['mst_class'].replace(0,np.nan)
dm0_raw['soil_class'] = dm0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
dm0_raw['thz_class'] = dm0_raw['thz_class'].replace([9,10],8)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
dm0_raw = dm0_raw.fillna(method='ffill')

#Try two different methods to handle the data:
#1: eliminate the respective rows:
dm0_elim = dm0_raw.loc[dm0_raw['mechanized'] > -9]
#dm0_elim = dm0_elim.loc[dm0_elim['n_fertilizer'] > -9]

#2: use ffill for the NaN values
dm0_raw['mechanized'] = dm0_raw['mechanized'].replace(-9,np.nan)
#dm0_raw['n_fertilizer'] = dm0_raw['n_fertilizer'].replace(-90000,np.nan)
#dm0_raw['p_fertilizer'] = dm0_raw['p_fertilizer'].replace(-90000,np.nan)
#dm0_raw['n_total'] = dm0_raw['n_total'].replace(-90000,np.nan)
dm0_raw = dm0_raw.fillna(method='ffill')
dm0_raw['mechanized'] = dm0_raw['mechanized'].fillna(1)

#for logging, replace 0s in n_man with a value a magnitude smaller than the smallest
#real value
dm0_flog = dm0_raw
dm0_flog['n_manure'] = dm0_flog['n_manure'].replace(0,0.00000000001)
dm0_elog = dm0_elim
dm0_elog['n_manure'] = dm0_elog['n_manure'].replace(0,0.00000000001)

###############################################################################
############Loading log transformed values for all variables##################
##############################################################################


#using log values for the input into the regression
#unfortunately the ln of 0 is not defined
#just keeping the 0 would skew the results as that would imply a 1 in the data when there is a 0
#could just use the smallest value of the dataset as a substitute?
data_log = {"yield": np.log(dm0_flog.loc[:,'yield']),
#		"n_fertilizer": np.log(dm0_flog.loc[:,'n_fertilizer']),
#		"p_fertilizer": np.log(dm0_flog.loc[:,'p_fertilizer']),
        "n_manure": np.log(dm0_flog.loc[:,'n_manure']),
#        "n_total" : np.log(dm0_flog.loc[:,'n_total']),
#        "pesticides_H": np.log(dm0_flog.loc[:,'pesticides_H']),
        "mechanized": dm0_flog.loc[:,'mechanized'],
#        "irrigation": np.log(irrigation.loc[:,'area']),
        "thz_class" : dm0_flog.loc[:,'thz_class'],
        "mst_class" : dm0_flog.loc[:,'mst_class'],
        "soil_class": dm0_flog.loc[:,'soil_class']
		}

dm0_log = pd.DataFrame(data=data_log)

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

##################Collinearity################################

###########RAW#################

#cautious, takes up a lot of memory
sb.pairplot(dmaize_dum_raw)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dmaize_cor_raw = dmaize_fit_raw.drop(['lat', 'lon', 'area', 'yield'], axis='columns')
#calculate spearman (rank transformed) correlation coeficcients between the 
#independent variables and save the values in a dataframe
sp = dmaize_cor_raw.corr(method='spearman')
print(sp)
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

X = add_constant(dmaize_cor_raw)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
#drop aggregated climate classes
cor_dum_raw = dmaize_cor_raw.drop(['thz_class', 'mst_class', 'soil_class'], axis='columns')
X1 = add_constant(cor_dum_raw)
pd.Series([variance_inflation_factor(X1.values, i) 
               for i in range(X1.shape[1])], 
              index=X1.columns)
#thz and mst factor levels are pretty highly correlated
#drop dummy variables
cor_clas_raw = dmaize_cor_raw.loc[:,['n_manure', 'mechanized', 'thz_class', 'mst_class', 
                                   'soil_class']]
X2 = add_constant(cor_clas_raw)
pd.Series([variance_inflation_factor(X2.values, i) 
               for i in range(X2.shape[1])], 
              index=X2.columns)

######################Regression##############################

#R-style formula
##############RAW#################
#rename Y variable
dmaize_fit_raw = dmaize_fit_raw.rename(columns={'yield':'Y'}, errors='raise')
dmaize_fit_elim = dmaize_fit_elim.rename(columns={'yield':'Y'}, errors='raise')
#determine models
#Normal distribution
mod_rawn = smf.ols(formula=' Y ~ n_manure + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dmaize_fit_raw)
mod_elimn = smf.ols(formula=' Y ~ n_manure + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dmaize_fit_elim)
#Gamma distribution
mod_rawg = smf.glm(formula='Y ~ n_manure + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_raw, 
              family=sm.families.Gamma())
mod_elimg = smf.glm(formula='Y ~ n_manure + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_elim, 
              family=sm.families.Gamma())
#Nullmodel
mod_raw0 = smf.glm(formula='Y ~ 1', data=dmaize_fit_raw, family=sm.families.Gamma())
mod_elim0 = smf.glm(formula='Y ~ 1', data=dmaize_fit_elim, family=sm.families.Gamma())
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
pseudoR_raw = 1-(269360/358300) #0.24823
pseudoR_elim = 1-(204560/289400) #0.29316
print(pseudoR_raw)
print(pseudoR_elim)

##############LOG################################
#rename Y variable
dmaize_fit_log= dmaize_fit_log.rename(columns={'yield':'Y'}, errors='raise')
#determine models
mod_logn = smf.ols(formula=' Y ~ n_manure + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dmaize_fit_log)           
#Fit models
fit_logn = mod_logn.fit()
#print results
print(fit_logn.summary())

   
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


