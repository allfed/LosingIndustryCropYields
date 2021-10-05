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

#import yield geopandas data for rice

rice_yield=pd.read_csv(params.geopandasDataDir + 'RICECropYieldHighRes.csv')

#select all rows from rice_yield for which the column growArea has a value greater than zero
rice_nozero=rice_yield.loc[rice_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
rice_kgha=rice_nozero['yield_kgPerHa']

rice_kgha_log=np.log(rice_kgha)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot rice yield distribution in a histogram
plt.hist(rice_kgha, bins=50)
plt.title('rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yield_kgPerHa
plt.hist(rice_kgha_log, bins=50)

'''
Fitting of distributions to the data and comparing the fit
'''

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

'''
#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_listr, pdf_listr, param_dictr)

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
datar_raw = {"lat": rice_yield.loc[:,'lats'],
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
drice_raw = pd.DataFrame(data=datar_raw)
#select only the rows where the area of the cropland is larger than 0
dr0_raw=drice_raw.loc[drice_raw['area'] > 0]

dr0_raw['pesticides_H'] = dr0_raw['pesticides_H'].replace(np.nan, -9)
dr0_raw['irrigation_rel'] = dr0_raw['irrigation_rel'].replace(np.nan, -9)

#test mech dataset values
r_test_Rech0 = dr0_raw.loc[dr0_raw['mechanized'] == 0] #134976
r_test_Rech1 = dr0_raw.loc[dr0_raw['mechanized'] == 1] #132231
r_test_Rechn = dr0_raw.loc[dr0_raw['mechanized'] == -9] #41358
#this is a problem: -9 is used as NaN value and there are way, way too many

r_test_f = dr0_raw.loc[dr0_raw['n_fertilizer'] < 0] #15974 0s, 7040 NaNs
r_test_pf = dr0_raw.loc[dr0_raw['p_fertilizer'] < 0] #20070 0s, 7040 NaNs 
r_test_Ran = dr0_raw.loc[dr0_raw['n_manure'] < 0] #17699 0s, no NaNs
r_test_p = dr0_raw.loc[dr0_raw['pesticides_H'] < 0] #no 0s, 130979 NaNs

dr0_raw['thz_class'] = dr0_raw['thz_class'].replace(0,np.nan)
dr0_raw['mst_class'] = dr0_raw['mst_class'].replace(0,np.nan)
dr0_raw['soil_class'] = dr0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
dr0_raw['thz_class'] = dr0_raw['thz_class'].replace([8,9,10],7)
dr0_raw['mst_class'] = dr0_raw['mst_class'].replace(2,1)
dr0_raw['mst_class'] = dr0_raw['mst_class'].replace(7,6)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
dr0_raw = dr0_raw.fillna(method='ffill')

#Handle the data by eliminating the rows without data:
dr0_elim = dr0_raw.loc[dr0_raw['pesticides_H'] > -9]
dr0_elim = dr0_elim.loc[dr0_elim['mechanized'] > -9] 

est_mechn = dr0_elim.loc[dr0_elim['n_fertilizer'] < 0] #98798

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
dr0_elim.loc[dr0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
dr0_elim.loc[dr0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
dr0_elim = dr0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
dr0_elim.loc[dr0_elim['n_total'] < 0, 'n_total'] = dr0_elim['n_fertilizer'] + dr0_elim['n_manure']

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
dr0_qt = dr0_elim.quantile([.1, .25, .5, .75, .8, .95, .999,.9999])
dr0_qt.reset_index(inplace=True, drop=True)
dr0_elim = dr0_elim.loc[dr0_elim['Y'] < dr0_qt.iloc[6,3]]
dr0_elim = dr0_elim.loc[dr0_elim['n_fertilizer'] < dr0_qt.iloc[6,4]]
dr0_elim = dr0_elim.loc[dr0_elim['p_fertilizer'] < dr0_qt.iloc[6,5]]
dr0_elim = dr0_elim.loc[dr0_elim['n_manure'] < dr0_qt.iloc[6,6]]
dr0_elim = dr0_elim.loc[dr0_elim['n_man_prod'] < dr0_qt.iloc[6,7]]
dr0_elim = dr0_elim.loc[dr0_elim['n_total'] < dr0_qt.iloc[6,8]]
dr0_elim = dr0_elim.loc[dr0_elim['pesticides_H'] < dr0_qt.iloc[6,9]]

plt.hist(dr0_elim['thz_class'], bins=50)

###############################################################################
##################################Outliers####################################
##############################################################################

#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('dr0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=dr0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=dr0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=dr0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=dr0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=dr0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=dr0_elim, x='Y')


ax = sb.boxplot(x=dr0_elim["irrigation_tot"])
ax = sb.boxplot(x=dr0_elim["irrigation_rel"])
ax = sb.boxplot(x="mechanized", y='Y', data=dr0_elim)
ax = sb.boxplot(x="thz_class", y='Y', hue='mechanized', data=dr0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=dr0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=dr0_elim)

#############################Get Dummies#####################

#mst, thz and soil are categorical variables which need to be converted into dummy variables before running the regression
dur_mst_elim = pd.get_dummies(dr0_elim['mst_class'])
dur_thz_elim = pd.get_dummies(dr0_elim['thz_class'])
dur_soil_elim = pd.get_dummies(dr0_elim['soil_class'])
#rename the columns according to the classes
dur_mst_elim = dur_mst_elim.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
dur_thz_elim = dur_thz_elim.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
dur_soil_elim = dur_soil_elim.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
drice_d_elim = pd.concat([dr0_elim, dur_mst_elim, dur_thz_elim, dur_soil_elim], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
drice_dur_elim = drice_d_elim.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
drice_val_elim = drice_dur_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
drice_fit_elim = drice_dur_elim.drop(drice_val_elim.index) #RAW

##################Collinearity################################

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
drice_cor_elim = drice_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(drice_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp_r = drice_cor_elim.corr(method='spearman')
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

Xr = add_constant(drice_cor_elim)
pd.Series([variance_inflation_factor(Xr.values, i) 
               for i in range(Xr.shape[1])], 
              index=Xr.columns)
'''
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

#determine models
#Normal distribution
r_mod_elimn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=drice_fit_elim)
#Gamma distribution
r_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=drice_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
r_mod_elim0 = smf.glm(formula='Y ~ 1', data=drice_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
r_fit_elimn = r_mod_elimn.fit()
r_fit_elimg = r_mod_elimg.fit()
r_fit_elim0 = r_mod_elim0.fit()
#print results
print(r_fit_elimn.summary()) #0.428
#LogLik: -1164400; AIC: 2329000; BIC: 2329000
print(r_fit_elimg.summary())
print(r_fit_elim0.summary())


###########Fit statistics#############
#calculate pseudo R² for the Gamma distribution
r_pseudoR_elim = 1-(40887/63577) #0.35689
print(r_pseudoR_elim)

#calculate AIC and BIC for Gamma
r_aic = r_fit_elimg.aic 
r_bic = r_fit_elimg.bic_llf
#LogLik: -118600; AIC: 2305951; BIC: 2306157 (-3305574: this doesn't track)

########Validation against the validation dataset########

#select the independent variables from the val dataset
r_val_elim = drice_val_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#fit the model against the validation data
r_pred_elim = r_fit_elimn.predict(r_val_elim)
r_pred_elimg = r_fit_elimg.predict(r_val_elim)

#calculate the R² scores
r2_score(drice_val_elim['Y'], r_pred_elim) #0.4239
r2_score(drice_val_elim['Y'], r_pred_elimg) #0.3776

#plot the predicted against the observed values
plt.scatter(r_pred_elim, drice_val_elim['Y'])
plt.scatter(r_pred_elimg, drice_val_elim['Y'])

#plot the histogram
plt.hist(r_pred_elimg, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

'''
plt.hist(drice_val_elim['Y'], bins=50)

an_elim = pd.concat([drice_val_elim, pred_elim, pred_elimg], axis='columns')
an_elim = an_elim.rename(columns={0:"pred_elim", 1:"pred_elimg"}, errors="raise")
sb.lmplot(x='pred_elimg', y='Y', data=an_elim)
'''



##########RESIDUALS#############

#######Gamma

#get the influence of the GLM model
r_stat_elimg = r_fit_elimg.get_influence()
#print(r_stat_elimg.summary_table()), there seems to be too much data
r_elimg_cook = pd.DataFrame(r_stat_elimg.cooks_distance).transpose()
r_elimg_cook = r_elimg_cook.rename(columns={0:"Cooks_d", 1:"ones"}, errors="raise")
r_data_infl = {'GLM_fitted': r_fit_elimg.fittedvalues,
       'hat_matrix':r_stat_elimg.hat_matrix_diag, 
       'resid_stud' : r_stat_elimg.resid_studentized}
r_elimg_infl = pd.DataFrame(data=r_data_infl).reset_index()
r_elimg_infl = pd.concat([r_elimg_infl, r_elimg_cook], axis='columns')
r_yiel_res =drice_fit_elim['Y'].reset_index(drop=True)

r_mod_resid = r_fit_elimg.resid_response.reset_index(drop=True)
r_mod_abs_resid = np.abs(r_mod_resid)
r_stud_sqrt = np.sqrt(np.abs(r_elimg_infl['resid_stud']))
r_resid = pd.concat([r_mod_resid, r_mod_abs_resid, r_stud_sqrt], axis='columns')
r_resid = r_resid.rename(columns={0:"resid_pear", 1:"resid_pear_abs", 'resid_stud':"resid_stud_sqrt"}, errors="raise")

r_elimg_infl_sample = pd.concat([r_elimg_infl, r_resid, r_yiel_res], axis='columns')
r_elimg_infl_sample = r_elimg_infl_sample.sample(frac=0.2, random_state=2705)

##########Residual Plot############
plot_elimg = plt.figure(4)
plot_elimg.set_figheight(8)
plot_elimg.set_figwidth(12)


plot_elimg.axes[0] = sb.residplot(r_elimg_infl['GLM_fitted'], drice_fit_elim['Y'], 
                          #lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_elimg.axes[0].set_title('Residuals vs Fitted')
plot_elimg.axes[0].set_xlabel('Fitted values')
plot_elimg.axes[0].set_ylabel('Residuals')

# annotations
r_abs_resid = r_mod_resid.sort_values(ascending=False)
r_abs_resid_top_3 = r_abs_resid[:3]

for i in r_abs_resid_top_3.index:
    plot_elimg.axes[0].annotate(i, 
                               xy=(r_elimg_infl['GLM_fitted'][i], 
                                   r_mod_resid[i]))

###############QQ-Plot########################

QQ = ProbPlot(r_elimg_infl['resid_stud'])
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
r_abs_norm_resid = np.flip(np.argsort(np.abs(r_elimg_infl['resid_stud'])), 0)
r_abs_norm_resid_top_3 = r_abs_norm_resid[:3]

for r, i in enumerate(r_abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   r_elimg_infl['resid_stud'][i]));

############Cook's distance plot##########

plot_lm_4 = plt.figure(4)
plot_lm_4.set_figheight(8)
plot_lm_4.set_figwidth(12)

plt.scatter(r_elimg_infl['hat_matrix'], r_elimg_infl['resid_stud'], alpha=0.5)
sb.regplot(r_elimg_infl['hat_matrix'], r_elimg_infl['resid_stud'], 
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
r_leverage_top_3 = np.flip(np.argsort(r_elimg_infl["Cooks_d"]), 0)[:3]

for i in r_leverage_top_3:
    plot_elimg.axes[0].annotate(i, 
                               xy=(r_elimg_infl['hat_matrix'][i], 
                                   r_elimg_infl['resid_stud'][i]))

# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(r_fit_elimg.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance') # 0.5 line
graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50)) # 1 line
plt.legend(loc='upper right');




'''
sm.graphics.plot_regress_exog(r_fit_elimg, 'n_total')
plt.show()


'''

#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

man_tot = fertilizer_man['applied_kgha'].loc[fertilizer_man['applied_kgha'] > -1].sum()
man_ptot = fertilizer_man['produced_kgha'].loc[fertilizer_man['produced_kgha'] > -1].sum()
per_man = man_tot/man_ptot *100

LoI_relim = dr0_elim.drop(['Y'], axis='columns')

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_relim['mechanized_y2'] = LoI_relim['mechanized'].replace(1,0)

#in year 1, there will probably be a slight surplus of N (production>application)
#divivde each cell of n_fertilizer with the sum of all cells and multiply with new total
LoI_relim['n_fert_y1'] = LoI_relim['n_fertilzer']/LoI_relim['n_fertilizer'].sum() * 14477
#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_relim.loc[LoI_relim['n_fert_y2'] > 0, 'n_fert_y1'] = 0
#LoI_mraw['n_fert_y1'] = LoI_mraw['n_fertilizer'].replace(1,0)

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_relim['labor'] = LoI_relim['area']/7.4 #current value is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer

#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_relim['area_LoI'] = LoI_relim['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations


LoI_relim['mechanized'] = LoI_relim['mechanized'].replace(1,0)
LoI_relim['labor'] = LoI_relim['area']/7.4


