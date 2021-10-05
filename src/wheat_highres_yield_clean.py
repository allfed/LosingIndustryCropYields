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
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.metrics import r2_score

params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for wheat

wheat_yield=pd.read_csv(params.geopandasDataDir + 'WHEACropYieldHighRes.csv')

#select all rows from wheat_yield for which the column growArea has a value greater than zero
wheat_nozero=wheat_yield.loc[wheat_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
wheat_kgha=wheat_nozero['yield_kgPerHa']

wheat_kgha_log=np.log(wheat_kgha)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot wheat yield distribution in a histogram
plt.hist(wheat_kgha, bins=50)
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=15000)

#plot log transformed values of yield_kgPerHa
plt.hist(wheat_kgha_log, bins=50)

#plots show that the raw values are right skewed so we try to fit a lognormal distribution and an exponentail distribution
#on the raw data and a normal distribution on the log transformed data

'''
Fitting of distributions to the data and comparing the fit
'''

'''
#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_listw, pdf_listw, param_dictw)

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

#test mech dataset values
w_test_mech0 = dw0_raw.loc[dw0_raw['mechanized'] == 0] #92565
w_test_mech1 = dw0_raw.loc[dw0_raw['mechanized'] == 1] #760670
w_test_mechn = dw0_raw.loc[dw0_raw['mechanized'] == -9] #98798
#this is a problem: -9 is used as NaN value and there are way, way too many

w_test_f = dw0_raw.loc[dw0_raw['n_fertilizer'] < 0] #19044 0s, 4512 NaNs
w_test_pf = dw0_raw.loc[dw0_raw['p_fertilizer'] < 0] #25889 0s, 4512 NaNs
w_test_man = dw0_raw.loc[dw0_raw['n_manure'] < 0] #12296 0s, 0 NaNs
w_test_p = dw0_raw.loc[dw0_raw['pesticides_H'] < 0] #no 0s, 120056 NaNs

dw0_raw['thz_class'] = dw0_raw['thz_class'].replace(0,np.nan)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(0,np.nan)
dw0_raw['soil_class'] = dw0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
dw0_raw['thz_class'] = dw0_raw['thz_class'].replace([8,9,10],7)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(2,1)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(7,6)

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

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
dw0_qt = dw0_elim.quantile([.1, .25, .5, .75, .8, .95, .999,.9999])
dw0_qt.reset_index(inplace=True, drop=True)
dw0_elim = dw0_elim.loc[dw0_elim['Y'] < dw0_qt.iloc[6,3]]
dw0_elim = dw0_elim.loc[dw0_elim['n_fertilizer'] < dw0_qt.iloc[6,4]]
dw0_elim = dw0_elim.loc[dw0_elim['p_fertilizer'] < dw0_qt.iloc[6,5]]
dw0_elim = dw0_elim.loc[dw0_elim['n_manure'] < dw0_qt.iloc[6,6]]
dw0_elim = dw0_elim.loc[dw0_elim['n_man_prod'] < dw0_qt.iloc[6,7]]
dw0_elim = dw0_elim.loc[dw0_elim['n_total'] < dw0_qt.iloc[6,8]]
dw0_elim = dw0_elim.loc[dw0_elim['pesticides_H'] < dw0_qt.iloc[6,9]]

#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('dw0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=dw0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=dw0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=dw0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=dw0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=dw0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=dw0_elim, x='Y')

ax = sb.boxplot(x=dw0_elim["irrigation_tot"])
ax = sb.boxplot(x=dw0_elim["irrigation_rel"])
ax = sb.boxplot(x="mechanized", y='Y', data=dw0_elim)
ax = sb.boxplot(x="thz_class", y='Y', data=dw0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=dw0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=dw0_elim)
plt.ylim(0,20000)

#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
duw_mst_elim = pd.get_dummies(dw0_elim['mst_class'])
duw_thz_elim = pd.get_dummies(dw0_elim['thz_class'])
duw_soil_elim = pd.get_dummies(dw0_elim['soil_class'])
#rename the columns according to the classes
duw_mst_elim = duw_mst_elim.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
duw_thz_elim = duw_thz_elim.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
duw_soil_elim = duw_soil_elim.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
dwheat_d_elim = pd.concat([dw0_elim, duw_mst_elim, duw_thz_elim, duw_soil_elim], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
dwheat_duw_elim = dwheat_d_elim.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dwheat_val_elim = dwheat_duw_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dwheat_fit_elim = dwheat_duw_elim.drop(dwheat_val_elim.index) #RAW

##################Collinearity################################

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dwheat_cor_elim = dwheat_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(dwheat_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp_w = dwheat_cor_elim.corr(method='spearman')
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

Xw = add_constant(dwheat_cor_elim)
pd.Series([variance_inflation_factor(Xw.values, i) 
               for i in range(Xw.shape[1])], 
              index=Xw.columns)
'''
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

######################Regression##############################

#R-style formula

link=sm.families.links.log

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
#LogLik: -1982100; AIC: 1938220; BIC: 1938423 (-3305574: this doesn't track)

########Validation against the validation dataset########

#select the independent variables from the val dataset
w_val_elim = dwheat_val_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#fit the model against the validation data
w_pred_elim = w_fit_elimn.predict(w_val_elim)
w_pred_elimg = w_fit_elimg.predict(w_val_elim)

#calculate the R² scores
r2_score(dwheat_val_elim['Y'], w_pred_elim) #0.3387
r2_score(dwheat_val_elim['Y'], w_pred_elimg) #0.3279

#plot the predicted against the observed values
plt.scatter(w_pred_elim, dwheat_val_elim['Y'])
plt.scatter(w_pred_elimg, dwheat_val_elim['Y'])

#plot the histogram
plt.hist(w_pred_elimg, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

'''
plt.hist(dwheat_val_elim['Y'], bins=50)

an_elim = pd.concat([dwheat_val_elim, pred_elim, pred_elimg], axis='columns')
an_elim = an_elim.rename(columns={0:"pred_elim", 1:"pred_elimg"}, errors="raise")
sb.lmplot(x='pred_elimg', y='Y', data=an_elim)
'''



##########RESIDUALS#############

#######Gamma

#get the influence of the GLM model
w_stat_elimg = w_fit_elimg.get_influence()
#print(w_stat_elimg.summary_table()), there seems to be too much data
w_elimg_cook = pd.DataFrame(w_stat_elimg.cooks_distance).transpose()
w_elimg_cook = w_elimg_cook.rename(columns={0:"Cooks_d", 1:"ones"}, errors="raise")
w_data_infl = {'GLM_fitted': w_fit_elimg.fittedvalues,
       'hat_matrix':w_stat_elimg.hat_matrix_diag, 
       'resid_stud' : w_stat_elimg.resid_studentized}
w_elimg_infl = pd.DataFrame(data=w_data_infl).reset_index()
w_elimg_infl = pd.concat([w_elimg_infl, w_elimg_cook], axis='columns')
w_yiel_res =dwheat_fit_elim['Y'].reset_index(drop=True)

w_mod_resid = w_fit_elimg.resid_response.reset_index(drop=True)
w_mod_abs_resid = np.abs(w_mod_resid)
w_stud_sqrt = np.sqrt(np.abs(w_elimg_infl['resid_stud']))
w_resid = pd.concat([w_mod_resid, w_mod_abs_resid, w_stud_sqrt], axis='columns')
w_resid = w_resid.rename(columns={0:"resid_pear", 1:"resid_pear_abs", 'resid_stud':"resid_stud_sqrt"}, errors="raise")

w_elimg_infl_sample = pd.concat([w_elimg_infl, w_resid, w_yiel_res], axis='columns')
w_elimg_infl_sample = w_elimg_infl_sample.sample(frac=0.2, random_state=2705)

##########Residual Plot############
plot_elimg = plt.figure(4)
plot_elimg.set_figheight(8)
plot_elimg.set_figwidth(12)


plot_elimg.axes[0] = sb.residplot(w_elimg_infl['GLM_fitted'], dwheat_fit_elim['Y'], 
                          #lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_elimg.axes[0].set_title('Residuals vs Fitted')
plot_elimg.axes[0].set_xlabel('Fitted values')
plot_elimg.axes[0].set_ylabel('Residuals')

# annotations
w_abs_resid =w_mod_resid.sort_values(ascending=False)
w_abs_resid_top_3 = w_abs_resid[:3]

for i in w_abs_resid_top_3.index:
    plot_elimg.axes[0].annotate(i, 
                               xy=(w_elimg_infl['GLM_fitted'][i], 
                                   w_mod_resid[i]))

###############QQ-Plot########################

QQ = ProbPlot(w_elimg_infl['resid_stud'])
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
w_abs_norm_resid = np.flip(np.argsort(np.abs(w_elimg_infl['resid_stud'])), 0)
w_abs_norm_resid_top_3 = w_abs_norm_resid[:3]

for r, i in enumerate(w_abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   w_elimg_infl['resid_stud'][i]));

############Cook's distance plot##########

plot_lm_4 = plt.figure(4)
plot_lm_4.set_figheight(8)
plot_lm_4.set_figwidth(12)

plt.scatter(w_elimg_infl['hat_matrix'], w_elimg_infl['resid_stud'], alpha=0.5)
sb.regplot(w_elimg_infl['hat_matrix'], w_elimg_infl['resid_stud'], 
            scatter=False, 
            ci=False, 
            #lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


plot_lm_4.axes[0].set_xlim(0, 0.004)
plot_lm_4.axes[0].set_ylim(-3, 21)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

# annotations
w_leverage_top_3 = np.flip(np.argsort(w_elimg_infl["Cooks_d"]), 0)[:3]

for i in w_leverage_top_3:
    plot_elimg.axes[0].annotate(i, 
                               xy=(w_elimg_infl['hat_matrix'][i], 
                                   w_elimg_infl['resid_stud'][i]))

# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(w_fit_elimg.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance') # 0.5 line
graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50)) # 1 line
plt.legend(loc='upper right');




'''
sm.graphics.plot_regress_exog(w_fit_elimg, 'n_total')
plt.show()


'''

#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

man_tot = fertilizer_man['applied_kgha'].loc[fertilizer_man['applied_kgha'] > -1].sum()
man_ptot = fertilizer_man['produced_kgha'].loc[fertilizer_man['produced_kgha'] > -1].sum()
per_man = man_tot/man_ptot *100

LoI_welim = dw0_elim.drop(['Y'], axis='columns')

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_welim['mechanized_y2'] = LoI_welim['mechanized'].replace(1,0)

#in year 1, there will probably be a slight surplus of N (production>application)
#divivde each cell of n_fertilizer with the sum of all cells and multiply with new total
LoI_welim['n_fert_y1'] = LoI_welim['n_fertilzer']/LoI_welim['n_fertilizer'].sum() * 14477
#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_welim.loc[LoI_welim['n_fert_y2'] > 0, 'n_fert_y1'] = 0
#LoI_mraw['n_fert_y1'] = LoI_mraw['n_fertilizer'].replace(1,0)

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_welim['labor'] = LoI_welim['area']/7.4 #current value is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer

#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_welim['area_LoI'] = LoI_welim['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations


LoI_welim['mechanized'] = LoI_welim['mechanized'].replace(1,0)
LoI_welim['labor'] = LoI_welim['area']/7.4


