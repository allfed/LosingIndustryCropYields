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
from src import outdoor_growth
from src.outdoor_growth import OutdoorGrowth
from src import stat_ut
import pandas as pd
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
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import GLMInfluence
from statsmodels.tools.tools import add_constant
from sklearn.metrics import r2_score

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
irr_t=pd.read_csv(params.geopandasDataDir + 'FracIrrigationAreaHighRes.csv')
crop = pd.read_csv(params.geopandasDataDir + 'FracCropAreaHighRes.csv')
irr_rel=pd.read_csv(params.geopandasDataDir + 'FracReliantHighRes.csv')
tillage=pd.read_csv(params.geopandasDataDir + 'TillageHighResAllCrops.csv')
aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')

#fraction of irrigation total is of total cell area so I have to divide it by the
#fraction of crop area in a cell and set all values >1 to 1
irr_tot = irr_t['fraction']/crop['fraction']
irr_tot.loc[irr_tot > 1] = 1
#dividing by 0 leaves a NaN value, so I have them all back to 0
irr_tot.loc[irr_tot.isna()] = 0

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
        "irrigation_tot": irr_tot,
        "irrigation_rel": irr_rel.loc[:,'frac_reliant'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
		}

#arrange data_raw in a dataframe
dmaize_raw = pd.DataFrame(data=datam_raw)
#select only the rows where the area of the cropland is larger than 0
dm0_raw=dmaize_raw.loc[dmaize_raw['area'] > 0]

dm0_raw['pesticides_H'] = dm0_raw['pesticides_H'].replace(np.nan, -9)
dm0_raw['irrigation_rel'] = dm0_raw['irrigation_rel'].replace(np.nan, -9)

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

#Handle the data by eliminating the rows without data:
dm0_elim = dm0_raw.loc[dm0_raw['pesticides_H'] > -9]
dm0_elim = dm0_elim.loc[dm0_raw['mechanized'] > -9] 
#replace remaining no data values in the fertilizer datasets with NaN and then fill them
dm0_elim.loc[dm0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
dm0_elim.loc[dm0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
dm0_elim = dm0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
dm0_elim.loc[dm0_elim['n_total'] < 0, 'n_total'] = dm0_elim['n_fertilizer'] + dm0_elim['n_manure']

###############Outliers###########################
m_out_f = dm0_elim.loc[dm0_elim['n_fertilizer'] > 400] #only 78 left
m_out_p = dm0_elim.loc[dm0_elim['p_fertilizer'] > 100] #169
m_out_man = dm0_elim.loc[dm0_elim['n_manure'] > 250] #35; 69 bei 200
m_out_prod = dm0_elim.loc[dm0_elim['n_man_prod'] > 1000] #32
m_out_n = dm0_elim.loc[(dm0_elim['n_manure'] > 250) | (dm0_elim['n_fertilizer'] > 400)] #has to be 78+35-1=112

#Boxplot of all the variables
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
ax = sb.boxplot(x="mst_class", y='Y', data=dm0_elim)
ax = sb.boxplot(x="soil_class", y='Y', data=dm0_elim)

#replace nonsense values in fertilizer and manure datasets
dm0_elim.loc[dm0_elim['n_fertilizer'] > 400, 'n_fertilizer'] = np.nan
dm0_elim.loc[dm0_elim['p_fertilizer'] > 100, 'p_fertilizer'] = np.nan
dm0_elim.loc[dm0_elim['n_manure'] > 250, 'n_manure'] = np.nan
#dm0_elim.loc[dm0_elim['n_man_prod'] > 1000, 'n_man_prod'] = np.nan
dm0_elim = dm0_elim.fillna(method='ffill')
dm0_elim['n_total'] = dm0_elim['n_manure'] + dm0_elim['n_fertilizer']


#mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########
mdum_mst = pd.get_dummies(dm0_elim['mst_class'])
mdum_thz = pd.get_dummies(dm0_elim['thz_class'])
mdum_soil = pd.get_dummies(dm0_elim['soil_class'])
#####Rename Columns##########
mdum_mst = mdum_mst.rename(columns={1:"LGP<60days", 2:"60-120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270-365days", 7:"365+days"}, errors="raise")
mdum_thz = mdum_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 
                        5:"Sub-trop_cool", 6:"Temp_mod", 7:"Temp_cool", 8:"Bor+Arctic"}, errors="raise")
mdum_soil = mdum_soil.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
dmaize_d_elim = pd.concat([dm0_elim, mdum_mst, mdum_thz, mdum_soil], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
dmaize_dum_elim = dmaize_d_elim.drop(['365+days','Bor+Arctic', 'L1_irr'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dmaize_val_elim = dmaize_dum_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dmaize_fit_elim = dmaize_dum_elim.drop(dmaize_val_elim.index) #RAW
##################Collinearity################################

#caution takes up a lot of memory
sb.pairplot(dmaize_fit_elim)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dmaize_cor_elim = dmaize_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
#calculate spearman (rank transformed) correlation coeficcients between the 
#independent variables and save the values in a dataframe
sp = dmaize_cor_elim.corr(method='spearman')
print(sp)
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

X = add_constant(dmaize_cor_elim)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
#drop aggregated climate classes
cor_elim = dmaize_cor_elim.drop(['270-365days', 'Trop_low'], axis='columns')
cor_elim = dmaize_cor_elim.drop(['LGP<60days', '60-120days', '120-180days',
                                    '180-225days', '225-270days', 
                                    '270-365days'], axis='columns')
X1 = add_constant(cor_elim)
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
'''
#rename Y variable
dmaize_fit_raw = dmaize_fit_raw.rename(columns={'yield':'Y'}, errors='raise')
dmaize_fit_elim = dmaize_fit_elim.rename(columns={'yield':'Y'}, errors='raise')
'''

#determine models
#Normal distribution
mod_elimn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dmaize_fit_elim)
#Gamma distribution
mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
mod_elim0 = smf.glm(formula='Y ~ 1', data=dmaize_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
fit_elimn = mod_elimn.fit()
fit_elimg = mod_elimg.fit()
fit_elim0 = mod_elim0.fit()
#print results
print(fit_elimn.summary())
print(fit_elimg.summary())
print(fit_elim0.summary())

#calculate pseudo R
pseudoR_elim = 1-(124770/200530) #0.29316 0.0982 0.348 0.366 0.37779
print(pseudoR_elim)

val_elim = dmaize_val_elim.drop(['lat', 'lon', 'area', 'Y', 'n_manure', 'n_fertilizer'], axis='columns')

pred_elim = fit_elimn.predict(val_elim)
#glm: like this they aren't valid -> link function
pred_elimg = fit_elimg.predict(val_elim)

r2_score(dmaize_val_elim['Y'], pred_elim) #0.34027102028441003
r2_score(dmaize_val_elim['Y'], pred_elimg) 

plt.scatter(pred_elim, dmaize_val_elim['Y'])
plt.scatter(pred_elimg, dmaize_val_elim['Y'])


plt.scatter(dmaize_val_elim['n_total'], pred_elim)
plt.scatter(dmaize_val_elim['n_total'], dmaize_val_elim['Y'])
plt.scatter(dmaize_val_elim['n_total'], pred_elimg)

an_raw = pd.concat([dmaize_val_raw, pred_raw, pred_rawg], axis='columns')
an_raw = an_raw.rename(columns={0:"pred_raw", 1:"pred_rawg"}, errors="raise")
sb.lmplot(x='pred_raw', y='Y', data=an_raw)

plt.hist(pred_elimg, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')


'''   
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
'''
##########RESIDUALS#############

stat_elimnorm = OLSInfluence(fit_elimn)
stat_elimn = fit_elimn.get_influence()
print(stat_elimn.summary_table())
lol = stat_elimn.cooks_distance
elimn_infl = pd.DataFrame(stat_elimn.cooks_distance).transpose()
elimn_infl = elimn_infl.rename(columns={0:"Cook's distance", 1:"ones"}, errors="raise")
elimn_hat = pd.Series(stat_elimn.hat_matrix_diag)
elimn_infl = pd.concat([elimn_infl, elimn_hat], axis='columns')
elimn_infl = elimn_infl.rename(columns={0:"hat matrix"}, errors="raise")

plt.scatter(elimn_infl['hat matrix'], elimn_infl["Cook's distance"])

print(elimn_infl.loc[(elimn_infl["Cook's distance"]>= 0.012) | (elimn_infl['hat matrix'] >= 0.008)])
'''
        Cook's distance  ones  hat matrix
29676          0.000327   1.0    0.012321
168756         0.012025   1.0    0.004417
'''
elim_new = dmaize_fit_elim.reset_index()
print(elim_new.loc[29676])
print(elim_new.loc[19052])

stat_elimg = fit_elimg.get_influence()
print(stat_elimg.summary_table())
lol = stat_elimg.resid_studentized
elimg_cook = pd.DataFrame(stat_elimg.cooks_distance).transpose()
elimg_cook = elimg_cook.rename(columns={0:"Cooks distance", 1:"ones"}, errors="raise")
dat = {'hat matrix':stat_elimg.hat_matrix_diag, 'resid_stud' : stat_elimg.resid_studentized}
elimg_infl = pd.DataFrame(data=dat)
elimg_infl = pd.concat([elimg_cook, elimg_infl], axis='columns')

plot_elimg = plt.figure(4)
plot_elimg.set_figheight(8)
plot_elimg.set_figwidth(12)

plt.scatter(elimg_infl['hat matrix'], elimg_infl['resid_stud'], alpha=0.5)
sb.regplot(elimg_infl['hat matrix'], elimg_infl['resid_stud'], 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_elimg.axes[0].set_xlim(0, 0.20)
plot_elimg.axes[0].set_ylim(-3, 5)
plot_elimg.axes[0].set_title('Residuals vs Leverage')
plot_elimg.axes[0].set_xlabel('Leverage')
plot_elimg.axes[0].set_ylabel('Standardized Residuals')

# annotations
leverage_top_3 = np.flip(np.argsort(elimg_infl["Cooks distance"]), 0)[:3]

for i in leverage_top_3:
    plot_elimg.axes[0].annotate(i, 
                               xy=(elimg_infl['hat matrix'][i], 
                                   elimg_infl['resid_stud'][i]))
    
# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(fit_elimg.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance') # 0.5 line
graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50)) # 1 line
plt.legend(loc='upper right');

stat_rawn1 = stat_rawn.summary_frame()

sm.graphics.plot_regress_exog(fit_rawn, 'n_total')
plt.show()

fitted_rawn = fit_rawn.fittedvalues
mod_res_alln_raw_fitted = mod_res_alln_raw.fittedvalues
mod_res_alln_log_fitted = mod_res_alln_log.fittedvalues

resd_rawn = fit_rawn.resid
mod_res_alln_raw_resd = mod_res_alln_raw.resid
mod_res_alln_log_resd = mod_res_alln_log.resid

modas_abs_resid = np.abs(modas1_resd)

plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sb.residplot(fitted_rawn, 'Y', data=dmaize_fit_raw, 
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


plt.scatter(fitted_rawn.resid_pearson)
sb.residplot(x=m_exog_n_log, y=m_endog_log)


#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

man_tot = fertilizer_man['applied_kgha'].loc[fertilizer_man['applied_kgha'] > -1].sum()
man_ptot = fertilizer_man['produced_kgha'].loc[fertilizer_man['produced_kgha'] > -1].sum()
per_man = man_tot/man_ptot *100

LoI_mraw = dm0_raw.drop(['Y'], axis='columns')
LoI_melim = dm0_elim.drop(['Y'], axis='columns')

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_mraw['mechanized_y2'] = LoI_mraw['mechanized'].replace(1,0)

#in year 1, there will probably be a slight surplus of N (production>application)
#divivde each cell of n_fertilizer with the sum of all cells and multiply with new total
LoI_mraw['n_fert_y1'] = LoI_mraw['n_fertilzer']/LoI_mraw['n_fertilizer'].sum() * 14477
#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_mraw.loc[LoI_mraw['n_fert_y2'] > 0, 'n_fert_y1'] = 0
#LoI_mraw['n_fert_y1'] = LoI_mraw['n_fertilizer'].replace(1,0)

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_mraw['labor'] = LoI_mraw['area']/7.4 #current value is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer

#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_mraw['area_LoI'] = LoI_mraw['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations


LoI_melim['mechanized'] = LoI_melim['mechanized'].replace(1,0)
LoI_melim['labor'] = LoI_melim['area']/7.4


