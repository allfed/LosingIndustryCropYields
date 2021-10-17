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
from src import utilities
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
from sklearn.metrics import d2_tweedie_score
from sklearn.metrics import mean_tweedie_deviance

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
#select only the rows where the area of the cropland is larger than 100 ha
dw0_raw=dwheat_raw.loc[dwheat_raw['area'] > 100]

dw0_raw['pesticides_H'] = dw0_raw['pesticides_H'].replace(np.nan, -9)
dw0_raw['irrigation_rel'] = dw0_raw['irrigation_rel'].replace(np.nan, -9)

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
dw0_qt = dw0_elim.quantile([.1, .25, .5, .75, .8, .95,.995, .999,.9999])
dw0_qt.reset_index(inplace=True, drop=True)
dw0_elim = dw0_elim.loc[dw0_elim['Y'] < dw0_qt.iloc[7,3]]
dw0_elim = dw0_elim.loc[dw0_elim['n_fertilizer'] < dw0_qt.iloc[7,4]]
dw0_elim = dw0_elim.loc[dw0_elim['p_fertilizer'] < dw0_qt.iloc[7,5]]
dw0_elim = dw0_elim.loc[dw0_elim['n_manure'] < dw0_qt.iloc[7,6]]
dw0_elim = dw0_elim.loc[dw0_elim['n_man_prod'] < dw0_qt.iloc[7,7]]
dw0_elim = dw0_elim.loc[dw0_elim['n_total'] < dw0_qt.iloc[7,8]]
dw0_elim = dw0_elim.loc[dw0_elim['pesticides_H'] < dw0_qt.iloc[7,9]]

#drop all rows with an area below 100 ha
#dw0_elim=dw0_elim.loc[dw0_elim['area'] > 100]

'''
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
'''
############################# Get Dummies #####################

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

############################# Validation ####################

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dwheat_val_elim = dwheat_duw_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dwheat_fit_elim = dwheat_duw_elim.drop(dwheat_val_elim.index) #RAW

################## Collinearity ################################

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

############ Variance inflation factor ##########################

Xw = add_constant(dwheat_cor_elim)
pd.Series([variance_inflation_factor(Xw.values, i) 
               for i in range(Xw.shape[1])], 
              index=Xw.columns)
'''
const                47.327215
p_fertilizer          5.614766
n_total               6.675488
pesticides_H          1.954755
mechanized            2.250868
irrigation_tot        2.603717
LGP<120days           2.557880
120-180days           2.898063
180-225days           2.404055
225-270days           1.849939
Trop_low              1.261454
Trop_high             1.221974
Sub-trop_warm         2.071177
Sub-trop_mod_cool     1.491580
Sub-trop_cool         1.495386
Temp_mod              1.450259
S1_very_steep         1.286700
S2_hydro_soil         1.468154
S3_no-slight_lim      4.258904
S4_moderate_lim       3.130808
S5_severe_lim         1.396076
dtype: float64

'''

###################### Regression ##############################

#R-style formula

link=sm.families.links.log

#determine models
w_mod_elimn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dwheat_fit_elim)
#Gamma distribution
w_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + irrigation_tot + mechanized + pesticides_H +  C(thz_class) + \
              C(mst_class) + C(soil_class)', data=dwheat_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
w_mod_elim0 = smf.glm(formula='Y ~ 1', data=dwheat_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
w_fit_elimn = w_mod_elimn.fit()
w_fit_elimg = w_mod_elimg.fit()
w_fit_elim0 = w_mod_elim0.fit()
#print results
#LogLik: -2021300; AIC: 4043000; BIC: 4043000
print(w_fit_elimn.summary())
print(w_fit_elimg.summary())
print(w_fit_elim0.summary())


table = sm.stats.anova_lm(w_fit_elimn, typ=2)
print(table)

###########Fit statistics#############
#calculate pseudo R² for the Gamma distribution
w_pseudoR_elim = 1-(46684/74377) #0.31124 0.3844
print(w_pseudoR_elim)

d2_tweedie_score(dwheat_fit_elim['Y'], w_fit_elimg.fittedvalues, power=2) #0.3843
np.sqrt(mean_tweedie_deviance(dwheat_fit_elim['Y'], w_fit_elimg.fittedvalues, power=2)) #0.5194
np.sqrt(mean_tweedie_deviance(dwheat_fit_elim['Y'], w_fit_elim0.fittedvalues, power=2)) #0.5194


d2_tweedie_score(dwheat_fit_elim['Y'], w_fit_elimn.fittedvalues, power=0) #0.4301
np.sqrt(mean_tweedie_deviance(dwheat_fit_elim['Y'], w_fit_elimn.fittedvalues, power=0)) #1451.013

#calculate AIC and BIC for Gamma
w_aic = w_fit_elimg.aic 
w_bic = w_fit_elimg.bic_llf
#LogLik: -1454600; AIC: 2909165; BIC: 2909366

########Validation against the validation dataset########

#select the independent variables from the val dataset
w_val_elim = dwheat_val_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#fit the model against the validation data
#w_pred_elim = w_fit_elimn.predict(w_val_elim)
w_pred_elimg = w_fit_elimg.predict(w_val_elim)

#calculate the R² scores
#r2_score(dwheat_val_elim['Y'], w_pred_elim) #0.3387
r2_score(dwheat_val_elim['Y'], w_pred_elimg) #0.3874

#plot the predicted against the observed values
#plt.scatter(w_pred_elim, dwheat_val_elim['Y'])
plt.scatter(w_pred_elimg, dwheat_val_elim['Y'])

#plot the histogram
plt.hist(w_pred_elimg, bins=50)
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

'''
plt.hist(dwheat_val_elim['Y'], bins=50)

an_elim = pd.concat([dwheat_val_elim, pred_elim, pred_elimg], axis='columns')
an_elim = an_elim.rename(columns={0:"pred_elim", 1:"pred_elimg"}, errors="raise")
sb.lmplot(x='pred_elimg', y='Y', data=an_elim)
'''




##########RESIDUALS for the Gamma distribution#############


#select the independent variables from the fit dataset
w_fit_elim = dwheat_fit_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#get the influence of the GLM model
w_stat_elimg = w_fit_elimg.get_influence()
#print(w_stat_elimg.summary_table()), there seems to be too much data

#store cook's distance in a variable
w_elimg_cook = pd.Series(w_stat_elimg.cooks_distance[0]).transpose()
w_elimg_cook = w_elimg_cook.rename("Cooks_d", errors="raise")

#store the actual yield, the fitted values on response and link scale, 
#the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
#the absolute value of the resp and the sqrt of the stud residuals in a dataframe
#reset the index but keep the old one as a column in order to combine the dataframe
#with Cook's distance
w_data_infl = { 'Yield': dwheat_fit_elim['Y'],
                'GLM_fitted': w_fit_elimg.fittedvalues,
               'Fitted_link': w_fit_elimg.predict(w_fit_elim, linear=True),
               'resid_pear': w_fit_elimg.resid_pearson, 
               'resid_stud' : w_stat_elimg.resid_studentized,
               'resid_resp_abs' : np.abs(w_fit_elimg.resid_response),
               'resid_stud_sqrt' : np.sqrt(np.abs(w_stat_elimg.resid_studentized)),
               'hat_matrix':w_stat_elimg.hat_matrix_diag}
w_elimg_infl = pd.DataFrame(data=w_data_infl).reset_index()
w_elimg_infl = pd.concat([w_elimg_infl, w_elimg_cook], axis='columns')


#take a sample of the influence dataframe to plot the lowess line
w_elimg_infl_sample = w_elimg_infl.sample(frac=0.1, random_state=2705)



##########Residual Plot############

#########Studentized residuals vs. fitted values on link scale######

plot_ws = plt.figure(4)
plot_ws.set_figheight(8)
plot_ws.set_figwidth(12)

plot_ws.axes[0] = sb.regplot('Fitted_link', 'resid_stud', data=w_elimg_infl_sample, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
#plt.scatter('Fitted_link', 'resid_stud', data=w_elimg_infl)
plot_ws.axes[0].set_title('Studentized Residuals vs Fitted on link scale')
plot_ws.axes[0].set_xlabel('Fitted values on link scale')
plot_ws.axes[0].set_ylabel('Studentized Residuals')

#########Response residuals vs. fitted values on response scale#######
plot_wr = plt.figure(4)
plot_wr.set_figheight(8)
plot_wr.set_figwidth(12)


plot_wr.axes[0] = sb.residplot('GLM_fitted', 'Yield', data=w_elimg_infl_sample, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_wr.axes[0].set_title('Residuals vs Fitted')
plot_wr.axes[0].set_xlabel('Fitted values')
plot_wr.axes[0].set_ylabel('Residuals')

# annotations
abs_resid = w_elimg_infl_sample['resid_resp_abs'].sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_wr.axes[0].annotate(i, 
                               xy=(w_elimg_infl_sample['GLw_fitted'][i], 
                                   w_elimg_infl_sample['resid_resp_abs'][i]))

###############QQ-Plot########################

QQ = ProbPlot(w_elimg_infl['resid_stud'])
plot_wq = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_wq.set_figheight(8)
plot_wq.set_figwidth(12)

plot_wq.axes[0].set_title('Normal Q-Q')
plot_wq.axes[0].set_xlabel('Theoretical Quantiles')
plot_wq.axes[0].set_ylabel('Standardized Residuals');

# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(w_elimg_infl['resid_stud'])), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_wq.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   w_elimg_infl['resid_stud'][i]));

############Cook's distance plot##########

#############Cook's distance vs. no of observation######

#sort cook's distance value to get the value for the largest distance####
w_cook_sort = w_elimg_cook.sort_values(ascending=False)
#select all Cook's distance values which are greater than 4/n (n=number of datapoints)
w_cook_infl = w_elimg_cook.loc[w_elimg_cook > (4/273772)].sort_values(ascending=False)

#barplot for values with the strongest influence (=largest Cook's distance)
#because running the function on all values takes a little longer
plt.bar(w_cook_infl.index, w_cook_infl)
plt.ylim(0, 0.01)

#plots for largest 3 cook values, the ones greater than 4/n and all distance values
plt.scatter(w_cook_infl.index[0:3], w_cook_infl[0:3])
plt.scatter(w_cook_infl.index, w_cook_infl)
plt.scatter(w_elimg_cook.index, w_elimg_cook)
plt.ylim(0, 0.01)

############Studentized Residuals vs. Leverage w. Cook's distance line#####

plot_wc = plt.figure(4)
plot_wc.set_figheight(8)
plot_wc.set_figwidth(12)

plt.scatter(w_elimg_infl_sample['hat_matrix'], w_elimg_infl_sample['resid_stud'], alpha=0.5)
sb.regplot(w_elimg_infl_sample['hat_matrix'], w_elimg_infl_sample['resid_stud'], 
            scatter=False, 
            ci=False, 
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


plot_wc.axes[0].set_xlim(0, 0.004)
plot_wc.axes[0].set_ylim(-3, 21)
plot_wc.axes[0].set_title('Residuals vs Leverage')
plot_wc.axes[0].set_xlabel('Leverage')
plot_wc.axes[0].set_ylabel('Standardized Residuals')

# annotate the three points with the largest Cooks distance value
leverage_top_3 = np.flip(np.argsort(w_elimg_infl_sample["Cooks_d"]), 0)[:3]

for i in leverage_top_3:
    plot_wc.axes[0].annotate(i, 
                               xy=(w_elimg_infl_sample['hat_matrix'][i], 
                                   w_elimg_infl_sample['resid_stud'][i]))

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
#index of leverage top 3 is not the index of the datapoints, therefore I print
#the w_elimg_infl rows at this index because it contains the old index as a column
for i in leverage_top_3:
    print(w_elimg_infl.iloc[i])

sm.graphics.plot_wegress_exog(w_fit_elimg, 'n_total')
plt.show()


'''
#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

#take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
#and available manure correctly
LoI_welim = dw0_raw

LoI_welim['mechanized'] = LoI_welim['mechanized'].replace(-9,np.nan)
LoI_welim['pesticides_H'] = LoI_welim['pesticides_H'].replace(-9,np.nan)

############ Mechanised ##########################

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_welim['mechanized_y2'] = LoI_welim['mechanized'].replace(1,0)

############ N fertilizer #########################

wn_drop= LoI_welim[((LoI_welim['mechanized'].isna())|(LoI_welim['pesticides_H'].isna()))
                & (LoI_welim['n_fertilizer']<0)].index
LoI_welim_pn = LoI_welim.drop(wn_drop)

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_welim_pn.loc[LoI_welim_pn['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
LoI_welim_pn.loc[LoI_welim_pn['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
LoI_welim_pn[['n_fertilizer','p_fertilizer']] = LoI_welim_pn[['n_fertilizer','p_fertilizer']].fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
LoI_welim_pn.loc[LoI_welim_pn['n_total'] < 0, 'n_total'] = LoI_welim_pn['n_fertilizer'] + LoI_welim_pn['n_manure']

#drop the nonsense values (99.9th percentile) in the n and p fertilizer columns
LoI_welim_pn = LoI_welim_pn.loc[LoI_welim_pn['n_fertilizer'] < dw0_qt.iloc[7,4]]#~180
LoI_welim_pn = LoI_welim_pn.loc[LoI_welim_pn['p_fertilizer'] < dw0_qt.iloc[7,5]] #~34

#in year 1, there will probably be a slight surplus of N (production>application)
#calculate kg N applied per cell
LoI_welim_pn['n_kg'] = LoI_welim_pn['n_fertilizer']*LoI_welim_pn['area']
#calculate the fraction of the total N applied to wheat fields for each cell
LoI_welim_pn['n_ffrac'] = LoI_welim_pn['n_kg']/(LoI_welim_pn['n_kg'].sum())

#calculate the fraction of total N applied to wheat fields of the total N applied
#divide total of wheat N by 1000000 to get from kg to thousand t
w_nfert_frac = (LoI_welim_pn['n_kg'].sum())/1000000/118763
#calculate the new total for N wheat in year one based on the N total surplus
w_ntot_new = w_nfert_frac * 14477 * 1000000

#calculate the new value of N application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_welim_pn['n_fert_y1'] = (w_ntot_new * LoI_welim_pn['n_ffrac']) / LoI_welim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_welim_pn['n_fert_y2'] = 0
#LoI_welim_pn.loc[LoI_welim_pn['n_fert_y2'] > 0, 'n_fert_y1'] = 0

############## P Fertilizer #####################

#in year 1, there will probably be a slight surplus of P (production>application)
#calculate kg p applied per cell
LoI_welim_pn['p_kg'] = LoI_welim_pn['p_fertilizer']*LoI_welim_pn['area']
#calculate the fraction of the total N applied to wheat fields for each cell
LoI_welim_pn['p_ffrac'] = LoI_welim_pn['p_kg']/(LoI_welim_pn['p_kg'].sum())

#calculate the fraction of total P applied to wheat fields on the total P applied to cropland
#divide total of wheat P by 1000000 to get from kg to thousand t
w_pfert_frac = (LoI_welim_pn['p_kg'].sum())/1000000/45858
#calculate the new total for P wheat in year one based on the P total surplus
w_ptot_new = w_pfert_frac * 4142 * 1000000

#calculate the new value of P application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_welim_pn['p_fert_y1'] = (w_ptot_new * LoI_welim_pn['p_ffrac']) / LoI_welim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_welim_pn['p_fert_y2'] = 0

############# N Manure ###################

#drop the rows containing nonsense values (99th percentile) in the manure column
LoI_welim_man = LoI_welim.loc[LoI_welim['n_manure'] < dw0_qt.iloc[7,6]] #~11

#calculate kg N applied per cell: 1,018,425,976.75 kg total
LoI_welim_man['man_kg'] = LoI_welim_man['n_manure']*LoI_welim_man['area']
#calculate the fraction of the total N applied to wheat fields for each cell
LoI_welim_man['n_mfrac'] = LoI_welim_man['man_kg']/(LoI_welim_man['man_kg'].sum())

#calculate the fraction of total N applied to wheat fields of the total N applied to cropland
#divide total of wheat N by 1000000 to get from kg to thousand t
w_nman_frac = (LoI_welim_man['man_kg'].sum())/1000000/24000

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_welim_man['labor'] = LoI_welim_man['area']/5 #current value (7.4) is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer
#I chose 5 now just because I don't believe 7.4 is correct

#calculate mean excretion rate of each cow in one year: cattle supplied ~ 43.7% of 131000 thousand t
#manure production in 2014, there were ~ 1.008.570.000(Statista)/1.439.413.930(FAOSTAT) 
#heads of cattle in 2014
cow_excr = 131000000000*0.437/1439413930

#calculate available manure based on cow labor demand: 1,278,868,812.065 kg
w_man_av = cow_excr * LoI_welim_man['labor'].sum()
#more manure avialable then currently applied, but that is good as N from mineral
#fertilizer will be missing

#calculate the new value of man application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_welim_man['man_fert'] = (w_man_av * LoI_welim_man['n_mfrac']) / LoI_welim_man['area']


########### N total ######################

LoI_welim['N_toty1'] = LoI_welim_pn['n_fert_y1'] + LoI_welim_man['man_fert']
#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_welim['area_LoI'] = LoI_welim['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations

############## Pesticides #####################

LoI_welimp = LoI_welim.loc[LoI_welim['pesticides_H'].notna()]
LoI_welimp = LoI_welimp.loc[LoI_welimp['pesticides_H'] < dw0_qt.iloc[7,9]]#~11
#in year 1, there will probably be a slight surplus of Pesticides (production>application)
#calculate kg p applied per cell
LoI_welimp['pest_kg'] = LoI_welimp['pesticides_H']*LoI_welimp['area']
#calculate the fraction of the total N applied to wheat fields for each cell
LoI_welimp['pest_frac'] = LoI_welimp['pest_kg']/(LoI_welimp['pest_kg'].sum())

#calculate the fraction of total pesticides applied to wheat fields on the total pesticides applied to cropland
#divide total of wheat pesticides by 1000 to get from kg to t
w_pest_frac = (LoI_welimp['pest_kg'].sum())/1000/4190985

#due to missing reasonable data on the pesticide surplus, it is assumed that the
#surplus is in the same range as for P and N fertilizer
frac_pest = ((14477/118763) + (4142/45858))/2
#calculate the new total for pesticides wheat in year one based on the pesticides total surplus
w_pestot_new = w_pest_frac * (4190985 * frac_pest) * 1000

#calculate the new value of pesticides application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_welimp['pest_y1'] = (w_pestot_new * LoI_welimp['pest_frac']) / LoI_welimp['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_welimp['pest_y2'] = 0


########## Irrigation ####################

#in LoI it is assumed that only irrigation which is not reliant on electricity
#can still be maintained
#calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
#(fraction of irrigated cropland reliant on electricity)
LoI_welim['irr_LoI'] = LoI_welim['irrigation_tot'] * (1- LoI_welim['irrigation_rel'])

###########Combine the different dataframes and drop rows with missing values#########

LoI_welim = pd.concat([LoI_welim, LoI_welim_pn['n_fert_y1'], LoI_welim_pn['n_fert_y2'],
                       LoI_welim_pn['p_fert_y1'], LoI_welim_pn['p_fert_y2'],
                       LoI_welim_man['man_fert'], LoI_welimp['pest_y1'], 
                       LoI_welimp['pest_y2']], axis='columns')

#Handle the data by eliminating the rows without data:
LoI_welim = LoI_welim.dropna()

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
#dw0_qt = dw0_elim.quantile([.1, .25, .5, .75, .8,.85, .87, .9, .95,.975, .99,.995, .999,.9999])
#dw0_qt.reset_index(inplace=True, drop=True)
LoI_welim = LoI_welim.loc[LoI_welim['Y'] < dw0_qt.iloc[7,3]] #~12500
#dw0_elim = dw0_elim.loc[dw0_elim['n_man_prod'] < dw0_qt.iloc[7,7]] #~44
LoI_welim = LoI_welim.loc[LoI_welim['n_total'] < dw0_qt.iloc[7,8]] #~195


#########################Prediction of LoI yields#########################

################## Year 1 ##################

#select the rows from LoI_welim which contain the independent variables for year 1
LoI_w_year1 = LoI_welim.iloc[:,[10,13,14,15,17,19,22,25]]
#reorder the columns according to the order in dw0_elim
LoI_w_year1 = LoI_w_year1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_w_year1 = LoI_w_year1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                      'pest_y1':"pesticides_H",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")
#predict the yield for year 1 using the gamma GLM
w_yield_y1 = w_fit_elimg.predict(LoI_w_year1)
#calculate the change rate from actual yield to the predicted yield
w_y1_change = ((w_yield_y1-wheat_kgha)/wheat_kgha).dropna()

#calculate statistics for yield and change rate

#yield
wmean_y1_weigh = round(np.average(w_yield_y1, weights=LoI_welim['area']),2) #2252.36kg/ha
wmax_y1 = w_yield_y1.max() #6610.19 kg/ha
wmin_y1 = w_yield_y1.min() #588.96 kg/ha

#change rate
wmean_y1c_weigh = np.average(w_y1_change, weights=LoI_welim['area']) #+0.037 (~+4%)
wmax_y1c = w_y1_change.max() # +40.39 (~+4000%)
wmin_y1c = w_y1_change.min() #-0.9330 (~-93%)

################## Year 2 ##################

#select the rows from LoI_welim which contain the independent variables for year 2
LoI_w_year2 = LoI_welim.iloc[:,[13,14,15,16,19,23,24,26]]
#reorder the columns according to the order in dw0_elim
LoI_w_year2 = LoI_w_year2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_w_year2 = LoI_w_year2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                      'pest_y2':"pesticides_H",'mechanized_y2':"mechanized",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")
#predict the yield for year 2 using the gamma GLM
w_yield_y2 = w_fit_elimg.predict(LoI_w_year2)
#calculate the change from actual yield to the predicted yield
w_y2_change = (w_yield_y2-wheat_kgha)/wheat_kgha
w_y2_change = w_y2_change.dropna()

#calculate statistics for yield and change rate

#yield
wmean_y2_weigh = round(np.average(w_yield_y2, weights=LoI_welim['area']),2) #1799.3kg/ha
wmax_y2 = w_yield_y2.max() #4515.31kg/ha
wmin_y2 = w_yield_y2.min() #579.24kg/ha

#change rate
wmean_y2c =round(np.average(w_y2_change, weights=LoI_welim['area']),2) #-0.17 (~-17%)
wmax_y2c = w_y2_change.max() #34.56 (~+3456%)
wmin_y2c = w_y2_change.min() #-0.9394 (~-94%)

#combine both yields and change rates with the latitude and longitude values
LoI_wheat = pd.concat([wheat_yield['lats'], wheat_yield['lons'], w_yield_y1,
                       w_y1_change, w_yield_y2, w_y2_change], axis='columns')
LoI_wheat = LoI_wheat.rename(columns={0:"w_yield_y1", 1:"w_y1_change", 
                                      2:"w_yield_y2",3:"w_y2_change"}, errors="raise")
#save the dataframe in a csv
LoI_wheat.to_csv(params.geopandasDataDir + "LoIWheatYieldHighRes.csv")


test = pd.concat([LoI_w_year2, dwheat_duw_elim[16:30]])

para_L = w_fit_elimg.conf_int()[0]
para_H = w_fit_elimg.conf_int()[1]
para_L = w_fit_elimg.params - 1.96*w_fit_elimg.bse
para_L1 = np.exp(w_fit_elimg.params) - np.exp(w_fit_elimg.bse)
para_H = w_fit_elimg.params + w_fit_elimg.bse
w_mod_elimg.predict(params=para_L, exog=LoI_w_year2)
(w_y2_change * LoI_welim['area']).sum()

round(w_y1_change.quantile([.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99]),2)
round(w_y2_change.quantile([.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99]),2)

w_y2_change.sum()/342277
round(np.average(dw0_elim['Y'], weights=dw0_elim['area']),2)
#Year 1 yield
2252.36/3123.0 #~72.1% of current average yield
(w_yield_y1 * LoI_welim['area']).sum()
443529453669.6886/674830448927.756 #~65.7% of current total yield
#Year 2 yield
1799.3/3123.0 #57.6% of current average yield
(w_yield_y2 * LoI_welim['area']).sum()
354314107844.2012/674830448927.756 #~52.5% of current total yield

utilities.create5minASCIIneg(LoI_wheat,'w_y1_change',params.asciiDir+'LoIWheatYieldChange_y1')
utilities.create5minASCIIneg(LoI_wheat,'w_yield_y1',params.asciiDir+'LoIWheatYield_y1')
utilities.create5minASCIIneg(LoI_wheat,'w_y2_change',params.asciiDir+'LoIWheatYieldChange_y2')
utilities.create5minASCIIneg(LoI_wheat,'w_yield_y2',params.asciiDir+'LoIWheatYield_y2')
