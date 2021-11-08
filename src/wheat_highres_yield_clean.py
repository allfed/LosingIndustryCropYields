'''

File containing the code to prepare the input data and perform a multiple regression
on yield for wheat at 5 arcmin resolution


Jessica Mörsdorf
jessica@allfed.info
jessica.m.moersdorf@umwelt.uni-giessen.de

'''

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
       sys.path.append(module_path)

from src import params
from src import utilities
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.metrics import d2_tweedie_score
from sklearn.metrics import mean_tweedie_deviance

params.importAll()


'''
Import yield data, extract zeros and plot the data
'''

# import yield data for wheat

wheat_yield = pd.read_csv(params.geopandasDataDir + 'WHEACropYieldHighRes.csv')

# select all rows from wheat_yield for which the column growArea has a value greater than zero
wheat_nozero = wheat_yield.loc[wheat_yield['growArea'] > 0]
# compile yield data where area is greater 0 in a new array
wheat_kgha = wheat_nozero['yield_kgPerHa']

round(np.average(wheat_kgha, weights=wheat_nozero['growArea']),2)

# sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

# plot wheat yield distribution in a histogram
plt.hist(wheat_kgha, bins=50)
plt.title('wheat yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')
plt.xlim(right=15000)


'''
Fitting of distributions to the data and comparing the fit
'''

#calculate loglik, AIC & BIC for each distribution
#st = stat_ut.stat_overview(dist_listw, pdf_listw, param_dictw)

#      Distribution  loglikelihood           AIC           BIC
#7  normal on log  -5.446560e+05  1.089328e+06  1.089416e+06
#6  Inverse Gamma  -4.059595e+06  8.119207e+06  8.119295e+06
#4     halfnormal  -4.080175e+06  8.160366e+06  8.160455e+06
#1    exponential  -4.102465e+06  8.204946e+06  8.205034e+06
#3         normal  -4.166714e+06  8.333444e+06  8.333532e+06
#0        lognorm  -5.009504e+06  1.001902e+07  1.001911e+07
#2        weibull  -5.153811e+06  1.030764e+07  1.030773e+07
#5          Gamma           -inf           inf           inf
#best fit so far: normal on log values by far, then Gamma on non-log


'''
Import factor datasets and extract zeros,
Harmonize units and correct irrigation fraction
'''
w_pesticides = pd.read_csv(params.geopandasDataDir + 'WheatPesticidesHighRes.csv')
fertilizer = pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv')  # kg/m²
fertilizer_man = pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv')  # kg/km²
irr_t = pd.read_csv(params.geopandasDataDir + 'FracIrrigationAreaHighRes.csv')
crop = pd.read_csv(params.geopandasDataDir + 'FracCropAreaHighRes.csv')
irr_rel = pd.read_csv(params.geopandasDataDir + 'FracReliantHighRes.csv')
tillage = pd.read_csv(params.geopandasDataDir + 'TillageHighResAllCrops.csv')
aez = pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')

#fraction of irrigation total is of total cell area so it has to be divided by the
#fraction of crop area in a cell and set all values >1 to 1
irr_tot = irr_t['fraction']/crop['fraction']
irr_tot.loc[irr_tot > 1] = 1
#dividing by 0 leaves a NaN value, have to be set back to 0
irr_tot.loc[irr_tot.isna()] = 0

#fertilizer is in kg/m² and fertilizer_man is in kg/km² while yield and pesticides are in kg/ha
#all continuous variables are transfowmed to kg/ha
n_new = fertilizer['n'] * 10000
p_new = fertilizer['p'] * 10000
fert_new = pd.concat([n_new, p_new], axis='columns')
fert_new.rename(columns={'n': 'n_kgha', 'p': 'p_kgha'}, inplace=True)
fertilizer = pd.concat([fertilizer, fert_new], axis='columns')  # kg/ha

applied_new = fertilizer_man['applied'] / 100
produced_new = fertilizer_man['produced'] / 100
man_new = pd.concat([applied_new, produced_new], axis='columns')
man_new.rename(columns={'applied': 'applied_kgha', 'produced': 'produced_kgha'}, inplace=True)
fertilizer_man = pd.concat([fertilizer_man, man_new], axis='columns')  # kg/ha

# compile a combined factor for N including both N from fertilizer and manure
N_total = fertilizer['n_kgha'] + fertilizer_man['applied_kgha']  # kg/ha


'''
Loading variables into a combined dataframe and preparing the input
data for analysis by filling/eliminating missing data points, deleting
outliers and combining levels of categorical factors
'''

dataw_raw = {"lat": wheat_yield.loc[:, 'lats'],
             "lon": wheat_yield.loc[:, 'lons'],
             "area": wheat_yield.loc[:, 'growArea'],
             "Y": wheat_yield.loc[:, 'yield_kgPerHa'],
             "n_fertilizer": fertilizer.loc[:, 'n_kgha'],
             "p_fertilizer": fertilizer.loc[:, 'p_kgha'],
             "n_manure": fertilizer_man.loc[:, 'applied_kgha'],
             "n_man_prod": fertilizer_man.loc[:, 'produced_kgha'],
             "n_total": N_total,
             "pesticides_H": w_pesticides.loc[:, 'total_H'],
             "mechanized": tillage.loc[:, 'is_mech'],
             "irrigation_tot": irr_tot,
             "irrigation_rel": irr_rel.loc[:, 'frac_reliant'],
             "thz_class": aez.loc[:, 'thz'],
             "mst_class": aez.loc[:, 'mst'],
             "soil_class": aez.loc[:, 'soil'],
             #"Y_log": np.log(wheat_yield.loc[:, 'yield_kgPerHa'])
             }

#arrange data_raw in a dataframe
dwheat_raw = pd.DataFrame(data=dataw_raw)
#select only the rows where the area of the cropland is larger than 100 ha
dw0_raw = dwheat_raw.loc[dwheat_raw['area'] > 100]

dw0_raw['pesticides_H'] = dw0_raw['pesticides_H'].replace(np.nan, -9)
dw0_raw['irrigation_rel'] = dw0_raw['irrigation_rel'].replace(np.nan, -9)

#replace 0s in the moisture, temperature and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
dw0_raw['thz_class'] = dw0_raw['thz_class'].replace(0, np.nan)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(0, np.nan)
dw0_raw['soil_class'] = dw0_raw['soil_class'].replace([0, 7, 8], np.nan)
#replace 8,9 & 10 with 7 in the temperature class to combine all three classes
#into one Temp,cool-Arctic class
#repalce 2 with 1 and 7 with 6 in the moisture class to compile them into one class each
dw0_raw['thz_class'] = dw0_raw['thz_class'].replace([8, 9, 10], 7)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(2, 1)
dw0_raw['mst_class'] = dw0_raw['mst_class'].replace(7, 6)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
dw0_raw = dw0_raw.fillna(method='ffill')

#Eliminate the rows without data:
dw0_elim = dw0_raw.loc[dw0_raw['pesticides_H'] > -9]
dw0_elim = dw0_elim.loc[dw0_raw['mechanized'] > -9]

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
#because there are only few left
dw0_elim.loc[dw0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan  # only 2304 left, so ffill
dw0_elim.loc[dw0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
dw0_elim = dw0_elim.fillna(method='ffill')
# replace no data values in n_total with the sum of the newly filled n_fertilizer and the
# n_manure values
dw0_elim.loc[dw0_elim['n_total'] < 0, 'n_total'] = dw0_elim['n_fertilizer'] + dw0_elim['n_manure']

#calculate the 25th, 50th, 75th, 85th, 95th, 99th and 99.9th percentile
dw0_qt = dw0_elim.quantile([.25, .5, .75, .85, .95, .99, .999])
dw0_qt.reset_index(inplace=True, drop=True)

#Values above the 99.9th percentile are considered unreasonable outliers
#Calculate number and statistic properties of the outliers
Y_out = dw0_elim.loc[dw0_elim['Y'] > dw0_qt.iloc[6, 3]]  # ~12500
nf_out = dw0_elim.loc[dw0_elim['n_fertilizer'] > dw0_qt.iloc[6, 4]]
pf_out = dw0_elim.loc[dw0_elim['p_fertilizer'] > dw0_qt.iloc[6, 5]]
nm_out = dw0_elim.loc[dw0_elim['n_manure'] > dw0_qt.iloc[5, 6]]
nt_out = dw0_elim.loc[dw0_elim['n_total'] > dw0_qt.iloc[6, 8]]  
P_out = dw0_elim.loc[dw0_elim['pesticides_H'] > dw0_qt.iloc[6, 9]]
w_out = pd.concat([Y_out['Y'], nf_out['n_fertilizer'], pf_out['p_fertilizer'],
                 nm_out['n_manure'], nt_out['n_total'], P_out['pesticides_H']], axis=1)
w_out.max()
w_out.min()
w_out.mean()

#Eliminate all points above the 99.9th percentile
dw0_elim = dw0_elim.loc[dw0_elim['Y'] < dw0_qt.iloc[6, 3]]
dw0_elim = dw0_elim.loc[dw0_elim['n_fertilizer'] < dw0_qt.iloc[6, 4]]
dw0_elim = dw0_elim.loc[dw0_elim['p_fertilizer'] < dw0_qt.iloc[6, 5]]
dw0_elim = dw0_elim.loc[dw0_elim['n_manure'] < dw0_qt.iloc[5, 6]]
dw0_elim = dw0_elim.loc[dw0_elim['n_man_prod'] < dw0_qt.iloc[6, 7]]
dw0_elim = dw0_elim.loc[dw0_elim['n_total'] < dw0_qt.iloc[6, 8]]
dw0_elim = dw0_elim.loc[dw0_elim['pesticides_H'] < dw0_qt.iloc[6, 9]]


'''
Dummy-code the categorical variables to be able to assess multicollinearity
'''

#mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########
duw_mst_elim = pd.get_dummies(dw0_elim['mst_class'])
duw_thz_elim = pd.get_dummies(dw0_elim['thz_class'])
duw_soil_elim = pd.get_dummies(dw0_elim['soil_class'])
#####Rename Columns##########
duw_mst_elim = duw_mst_elim.rename(columns={1: "LGP<120days", 3: "120-180days", 4: "180-225days",
                                            5: "225-270days", 6: "270+days"}, errors="raise")
duw_thz_elim = duw_thz_elim.rename(columns={1: "Trop_low", 2: "Trop_high", 3: "Sub-trop_warm", 4: "Sub-trop_mod_cool", 5: "Sub-trop_cool",
                                            6: "Temp_mod", 7: "Temp_cool+Bor+Arctic"}, errors="raise")
duw_soil_elim = duw_soil_elim.rename(columns={1: "S1_very_steep", 2: "S2_hydro_soil", 3: "S3_no-slight_lim", 4: "S4_moderate_lim",
                                              5: "S5_severe_lim", 6: "L1_irr"}, errors="raise")
# merge the two dummy dataframes with the rest of the variables
dwheat_d_elim = pd.concat([dw0_elim, duw_mst_elim, duw_thz_elim, duw_soil_elim], axis='columns')
#drop one column of each dummy (this value will be encoded by 0 in all columns)
dwheat_duw_elim = dwheat_d_elim.drop(['270+days', 'Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')


'''
Split the data into a validation and a calibration dataset
'''

# select a random sample of 20% from the dataset to set aside for later validation
# random_state argument ensures that the same sample is returned each time the code is run
dwheat_val_elim = dwheat_duw_elim.sample(frac=0.2, random_state=2705)  # RAW
# drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dwheat_fit_elim = dwheat_duw_elim.drop(dwheat_val_elim.index)


'''
Check for multicollinearity by calculating the two-way correlations and the VIF
'''

#extract lat, lon, area, yield, individual n columns, original climate class columns and irrigation for the LoI scenario
#from the fit dataset to test the correlations among the
#independent variables
dwheat_cor_elim = dwheat_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                        'irrigation_rel', 'thz_class',
                                        'mst_class', 'soil_class'], axis='columns')

#### Correlations ###

# calculates spearman (rank transfowmed) correlation coeficcients between the
# independent variables and saves the values in a dataframe
sp_w = dwheat_cor_elim.corr(method='spearman')

### Variance inflation factor ###

Xw = add_constant(dwheat_cor_elim)
pd.Series([variance_inflation_factor(Xw.values, i)
           for i in range(Xw.shape[1])],
          index=Xw.columns)

#const                47.327215
#p_fertilizer          5.614766
#n_total               6.675488
#pesticides_H          1.954755
#mechanized            2.250868
#irrigation_tot        2.603717
#LGP<120days           2.557880
#120-180days           2.898063
#180-225days           2.404055
#225-270days           1.849939
#Trop_low              1.261454
#Trop_high             1.221974
#Sub-trop_warm         2.071177
#Sub-trop_mod_cool     1.491580
#Sub-trop_cool         1.495386
#Temp_mod              1.450259
#S1_very_steep         1.286700
#S2_hydro_soil         1.468154
#S3_no-slight_lim      4.258904
#S4_moderate_lim       3.130808
#S5_severe_lim         1.396076


#######################################################################
########### Regression Calibration, Validation and Residuals###########
#######################################################################

'''
Calibrate the Regression model and calculate fit statistics
'''

link = sm.families.links.log

#determine model with a gamma distribution
w_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + irrigation_tot + mechanized + pesticides_H +  C(thz_class) + \
              C(mst_class) + C(soil_class)', data=dwheat_fit_elim,
                      family=sm.families.Gamma(link=sm.families.links.log))
# Nullmodel
w_mod_elim0 = smf.glm(formula='Y ~ 1', data=dwheat_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))

#Fit models
w_fit_elimg = w_mod_elimg.fit()
w_fit_elim0 = w_mod_elim0.fit()

#print results
print(w_fit_elimg.summary())
print(w_fit_elim0.summary())

#calculate the odds ratios on the response scale
np.exp(w_fit_elimg.params)

### Fit statistics ###

#calculate McFadden's roh² and the Root Mean Gamma Deviance (RMGD)
d2_tweedie_score(dwheat_fit_elim['Y'], w_fit_elimg.fittedvalues, power=2)  # 0.3778
np.sqrt(mean_tweedie_deviance(dwheat_fit_elim['Y'], w_fit_elimg.fittedvalues, power=2))  # 0.5194

# calculate AIC and BIC for Gamma
w_aic = w_fit_elimg.aic
w_bic = w_fit_elimg.bic_llf
#LogLik: -1454600; AIC: 2909165; BIC: 2909366


'''
Validate the model against the validation dataset
'''

#select the independent variables from the validation dataset
w_val_elim = dwheat_val_elim.iloc[:, [5, 8, 9, 10, 11, 13, 14, 15]]

#let the model predict yield values for the validation data
w_pred_elimg = w_fit_elimg.predict(w_val_elim)

#calculate McFadden's roh² and the RMGD scores
d2_tweedie_score(dwheat_val_elim['Y'], w_pred_elimg, power=2) #0.3707
np.sqrt(mean_tweedie_deviance(dwheat_val_elim['Y'], w_pred_elimg, power=2)) #0.5228


'''
Plot the Residuals for the model
'''
### Extract necessary measures ###

#select the independent variables from the fit dataset
w_fit_elim = dwheat_fit_elim.iloc[:, [5, 8, 9, 10, 11, 13, 14, 15]]

#get the influence of the GLM model
w_stat_elimg = w_fit_elimg.get_influence()

# store cook's distance in a variable
w_elimg_cook = pd.Series(w_stat_elimg.cooks_distance[0]).transpose()
w_elimg_cook = w_elimg_cook.rename("Cooks_d", errors="raise")

#store the actual yield, the fitted values on response and link scale, 
#the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
#the absolute value of the resp and the sqrt of the stud residuals in a dataframe
#reset the index but keep the old one as a column in order to combine the dataframe
#with Cook's distance
w_data_infl = {'Yield': dwheat_fit_elim['Y'],
               'GLM_fitted': w_fit_elimg.fittedvalues,
               'Fitted_link': w_fit_elimg.predict(w_fit_elim, linear=True),
               'resid_pear': w_fit_elimg.resid_pearson,
               'resid_stud': w_stat_elimg.resid_studentized,
               'resid_resp_abs': np.abs(w_fit_elimg.resid_response),
               'resid_stud_sqrt': np.sqrt(np.abs(w_stat_elimg.resid_studentized)),
               'hat_matrix': w_stat_elimg.hat_matrix_diag}
w_elimg_infl = pd.DataFrame(data=w_data_infl).reset_index()
w_elimg_infl = pd.concat([w_elimg_infl, w_elimg_cook], axis='columns')

# take a sample of the influence dataframe to plot the lowess line
w_elimg_infl_sample = w_elimg_infl.sample(frac=0.1, random_state=2705)

### Studentized residuals vs. fitted values on link scale ###

#set plot characteristics
plot_ws = plt.figure(4)
plot_ws.set_figheight(8)
plot_ws.set_figwidth(12)

#Draw a scatterplot of studentized residuals vs. fitted values on the link scale
#lowess=True draws a fitted line which shows if the relationship is linear
plot_ws.axes[0] = sb.regplot('Fitted_link', 'resid_stud', data=w_elimg_infl_sample,
                             lowess=True,
                             scatter_kws={'alpha': 0.5},
                             line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
#plt.scatter('Fitted_link', 'resid_stud', data=w_elimg_infl)

#plot labels
plot_ws.axes[0].set_title('Studentized Residuals vs Fitted on link scale')
plot_ws.axes[0].set_xlabel('Fitted values on link scale')
plot_ws.axes[0].set_ylabel('Studentized Residuals')

### Response residuals vs. fitted values on the response scale ###

#set plot characteristics
plot_wr = plt.figure(4)
plot_wr.set_figheight(8)
plot_wr.set_figwidth(12)

#Draw a scatterplot of response residuals vs. fitted values on the response scale
plot_wr.axes[0] = sb.residplot('GLM_fitted', 'Yield', data=w_elimg_infl_sample,
                               lowess=True,
                               scatter_kws={'alpha': 0.5},
                               line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#plot labels
plot_wr.axes[0].set_title('Residuals vs Fitted')
plot_wr.axes[0].set_xlabel('Fitted values')
plot_wr.axes[0].set_ylabel('Residuals')

#annotations of the three largest residuals
abs_resid = w_elimg_infl_sample['resid_resp_abs'].sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_wr.axes[0].annotate(i,
                             xy=(w_elimg_infl_sample['GLM_fitted'][i],
                                 w_elimg_infl_sample['resid_resp_abs'][i]))

### QQ-Plot for the studentized residuals ###

#Specifications of the QQ Plot
QQ = ProbPlot(w_elimg_infl['resid_stud'], dist=stats.gamma, fit=True)
plot_wq = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

#set plot characteristics
plot_wq.set_figheight(8)
plot_wq.set_figwidth(12)

#plot labels
plot_wq.axes[0].set_title('Normal Q-Q')
plot_wq.axes[0].set_xlabel('Theoretical Quantiles')
plot_wq.axes[0].set_ylabel('Standardized Residuals')

#annotations of the three largest residuals
abs_norm_resid = np.flip(np.argsort(np.abs(w_elimg_infl['resid_stud'])), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_wq.axes[0].annotate(i,
                             xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                 w_elimg_infl['resid_stud'][i]))

### Cook's distance plots ###

##Cook's distance vs. no of observation##

#sort cook's distance value to get the value for the largest distance####
w_cook_sort = w_elimg_cook.sort_values(ascending=False)
# select all Cook's distance values which are greater than 4/n (n=number of datapoints)
w_cook_infl = w_elimg_cook.loc[w_elimg_cook > (4/(168227-21))].sort_values(ascending=False)

#barplot for values with the strongest influence (=largest Cook's distance)
#because running the function on all values takes a little longer
plt.bar(w_cook_infl.index, w_cook_infl)
plt.ylim(0, 0.01)

#plots for the ones greater than 4/n and all distance values
plt.scatter(w_cook_infl.index, w_cook_infl)
plt.scatter(w_elimg_cook.index, w_elimg_cook)
plt.ylim(0, 0.01)

##Studentized Residuals vs. Leverage w. Cook's distance line##

#set plot characteristics
plot_wc = plt.figure(4)
plot_wc.set_figheight(8)
plot_wc.set_figwidth(12)

#Draw the scatterplott of the Studentized residuals vs. leverage
plt.scatter(w_elimg_infl_sample['hat_matrix'], w_elimg_infl_sample['resid_stud'], alpha=0.5)
sb.regplot(w_elimg_infl_sample['hat_matrix'], w_elimg_infl_sample['resid_stud'],
           scatter=False,
           ci=False,
           lowess=True,
           line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#plot boundaries and labels
plot_wc.axes[0].set_xlim(0, 0.004)
plot_wc.axes[0].set_ylim(-3, 21)
plot_wc.axes[0].set_title('Residuals vs Leverage')
plot_wc.axes[0].set_xlabel('Leverage')
plot_wc.axes[0].set_ylabel('Standardized Residuals')

#annotate the three points with the largest Cooks distance value
leverage_top_3 = np.flip(np.argsort(w_elimg_infl_sample["Cooks_d"]), 0)[:3]

for i in leverage_top_3:
    plot_wc.axes[0].annotate(i,
                             xy=(w_elimg_infl_sample['hat_matrix'][i],
                                 w_elimg_infl_sample['resid_stud'][i]))


###########################################################################
################ Loss of Industry Modelling ###############################
###########################################################################

'''
Prepare and modify datasets according to the assumptions of the LoI scenario
'''
# take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
# and available manure correctly
LoI_welim = dw0_raw

LoI_welim['mechanized'] = LoI_welim['mechanized'].replace(-9, np.nan)
LoI_welim['pesticides_H'] = LoI_welim['pesticides_H'].replace(-9, np.nan)

### Mechanised ###

#set mechanization to 0 in phase 2; due to the estimated stock in  fuel the variable remains 
#unchanged in phase 1
LoI_welim['mechanized_y2'] = LoI_welim['mechanized'].replace(1, 0)

### N fertilizer ###

#drop all cells where mechanized or pesticiedes AND n_fertilizer are no data values
wn_drop = LoI_welim[((LoI_welim['mechanized'].isna()) | (LoI_welim['pesticides_H'].isna()))
                    & (LoI_welim['n_fertilizer'] < 0)].index
LoI_welim_pn = LoI_welim.drop(wn_drop)

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_welim_pn.loc[LoI_welim_pn['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan
LoI_welim_pn.loc[LoI_welim_pn['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
LoI_welim_pn[['n_fertilizer', 'p_fertilizer']] = LoI_welim_pn[['n_fertilizer', 'p_fertilizer']].fillna(method='ffill')
# replace no data values in n_total with the sum of the newly filled n_fertilizer and the
# n_manure values
LoI_welim_pn.loc[LoI_welim_pn['n_total'] < 0, 'n_total'] = LoI_welim_pn['n_fertilizer'] + LoI_welim_pn['n_manure']

#drop the outliers (99.9th percentile) in the n and p fertilizer columns
LoI_welim_pn = LoI_welim_pn.loc[LoI_welim_pn['n_fertilizer'] < dw0_qt.iloc[6, 4]]
LoI_welim_pn = LoI_welim_pn.loc[LoI_welim_pn['p_fertilizer'] < dw0_qt.iloc[6, 5]]

#in phase 1, there will probably be a slight surplus of N (production>application)
#the surplus is assumed to be the new total

#calculate kg N applied per cell
LoI_welim_pn['n_kg'] = LoI_welim_pn['n_fertilizer']*LoI_welim_pn['area']
#calculate the fraction of the total N applied to wheat fields for each cell
LoI_welim_pn['n_ffrac'] = LoI_welim_pn['n_kg']/(LoI_welim_pn['n_kg'].sum())

#calculate the fraction of total N applied to wheat fields of the total N applied
#divide total of wheat N by 1000000 to get from kg to thousand t
w_nfert_frac = (LoI_welim_pn['n_kg'].sum())/1000000/118763
#calculate the new total for N wheat in phase one based on the N total surplus
w_ntot_new = w_nfert_frac * 14477 * 1000000

#calculate the new value of N application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_welim_pn['n_fert_y1'] = (w_ntot_new * LoI_welim_pn['n_ffrac']) / LoI_welim_pn['area']

#in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_welim_pn['n_fert_y2'] = 0

### P Fertilizer ###

#in phase 1, there will probably be a slight surplus of P (production>application)
#calculate kg p applied per cell
LoI_welim_pn['p_kg'] = LoI_welim_pn['p_fertilizer']*LoI_welim_pn['area']
#calculate the fraction of the total N applied to rice fields for each cell
LoI_welim_pn['p_ffrac'] = LoI_welim_pn['p_kg']/(LoI_welim_pn['p_kg'].sum())

#calculate the fraction of total P applied to wheat fields on the total P applied to cropland
#divide total of wheat P by 1000000 to get from kg to thousand t
w_pfert_frac = (LoI_welim_pn['p_kg'].sum())/1000000/45858
#calculate the new total for P wheat in phase one based on the P total surplus
w_ptot_new = w_pfert_frac * 4142 * 1000000

#calculate the new value of P application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_welim_pn['p_fert_y1'] = (w_ptot_new * LoI_welim_pn['p_ffrac']) / LoI_welim_pn['area']

#in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_welim_pn['p_fert_y2'] = 0

### N Manure ###

#drop the rows containing outliers (99th percentile) in the manure column
LoI_welim_man = LoI_welim.loc[LoI_welim['n_manure'] < dw0_qt.iloc[5, 6]]

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_welim_man['labor'] = LoI_welim_man['area']/5 #current value (7.4) is taken from Cole et al. (2016)
#due to information from a fawmer, the value is set at 5

# calculate mean excretion rate of each cow in one phase: cattle supplied ~ 43.7% of 131000 thousand t
# manure production in 2014, there were ~ 1.008.570.000(Statista)/1.439.413.930(FAOSTAT)
# heads of cattle in 2014
cow_excr = 131000000000*0.437/1439413930

# calculate the new value of man application rate in kg per ha per cell, according
# to the available cows in each cell due to labor demand
LoI_welim_man['man_fert'] = (cow_excr * LoI_welim_man['labor']) / LoI_welim_man['area']

#that leads the application rate being the same in every cell because everybody has the same number of cows per ha
#it's assumed to be the same for both phases


### N total ###

#in phase 1, the total N available is the sum of available fertilizer and manure
LoI_welim['N_toty1'] = LoI_welim_pn['n_fert_y1'] + LoI_welim_man['man_fert']

#in phase 2 there is no more artificial fertilizer, so N total is equal to man_fert

### Pesticides ###

#drop the cells containing NaN values and outliers
LoI_welimp = LoI_welim.loc[LoI_welim['pesticides_H'].notna()]
LoI_welimp = LoI_welimp.loc[LoI_welimp['pesticides_H'] < dw0_qt.iloc[6, 9]]

#in phase 1, there will probably be a slight surplus of Pesticides (production>application)

#calculate kg p applied per cell
LoI_welimp['pest_kg'] = LoI_welimp['pesticides_H']*LoI_welimp['area']
#calculate the fraction of the total N applied to wheat fields for each cell
LoI_welimp['pest_frac'] = LoI_welimp['pest_kg']/(LoI_welimp['pest_kg'].sum())

#calculate the fraction of total pesticides applied to wheat fields on the total pesticides applied to cropland
#divide total of wheat pesticides by 1000 to get from kg to t
w_pest_frac = (LoI_welimp['pest_kg'].sum())/1000/4190985

#due to missing reasonable data on the pesticide surplus, it is assumed that the
#surplus is in the same range as for P and N fertilizer
#the mean of N and P fertilizer surplus is calculated
frac_pest = ((14477/118763) + (4142/45858))/2
#calculate the new total for pesticides wheat in phase one based on the pesticides total surplus
w_pestot_new = w_pest_frac * (4190985 * frac_pest) * 1000

#calculate the new value of pesticides application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_welimp['pest_y1'] = (
    w_pestot_new * LoI_welimp['pest_frac']) / LoI_welimp['area']

# in phase 2 no industrially produced pesticides will be available anymore: set to 0
LoI_welimp['pest_y2'] = 0

### Irrigation ###

#in LoI it is assumed that only irrigation which is not reliant on electricity
#can still be maintained
#calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
#(fraction of irrigated cropland reliant on electricity)
LoI_welim['irr_LoI'] = LoI_welim['irrigation_tot'] * (1 - LoI_welim['irrigation_rel'])

### Combine the different dataframes and drop rows with missing values ###

LoI_welim = pd.concat([LoI_welim, LoI_welim_pn['n_fert_y1'], LoI_welim_pn['n_fert_y2'],
                       LoI_welim_pn['p_fert_y1'], LoI_welim_pn['p_fert_y2'],
                       LoI_welim_man['man_fert'], LoI_welimp['pest_y1'],
                       LoI_welimp['pest_y2']], axis='columns')

#Eliminate the rows without data:
LoI_welim = LoI_welim.dropna()

#Eliminating all points above the 99.9th percentile
LoI_welim = LoI_welim.loc[LoI_welim['Y'] < dw0_qt.iloc[6, 3]]
LoI_welim = LoI_welim.loc[LoI_welim['n_total'] < dw0_qt.iloc[6, 8]]


'''
Prediction of LoI yields and yield change rates in phase 1 and 2
'''
### Phase 1 ###

#select the rows from LoI_relim which contain the independent variables for phase 1
LoI_w_phase1 = LoI_welim.iloc[:, [10, 13, 14, 15, 17, 19, 22, 25]]
#reorder the columns according to the order in dw0_elim
LoI_w_phase1 = LoI_w_phase1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized',
                           'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_w_phase1 = LoI_w_phase1.rename(columns={'p_fert_y1': "p_fertilizer", 'N_toty1': "n_total",
                                          'pest_y1': "pesticides_H",
                                          'irr_LoI': "irrigation_tot"}, errors="raise")

#predict the yield for phase 1 using the gamma GLM
w_yield_y1 = w_fit_elimg.predict(LoI_w_phase1)
#calculate the change rate from actual yield to the predicted yield
w_y1_change = ((w_yield_y1-wheat_kgha)/wheat_kgha).dropna()
#calculate the number of cells with a postivie change rate
s1 = w_y1_change.loc[w_y1_change > 0]

#create a new variable with the yields for positive change rates set to orginial yields
w01 = w_y1_change.loc[w_y1_change > 0]
w_y1_0 = LoI_welim['Y']
w_y1_0 = w_y1_0[w01.index]
w011 = w_y1_change.loc[w_y1_change <= 0]
w_y1_1 = w_yield_y1[w011.index]
w_y1_y0 = w_y1_0.append(w_y1_1)

#calculate statistics for yield and change rate

#calculate weights for mean change rate calculation dependent on current yield
#and current maize area in a cell
ww=LoI_welim['Y']*dw0_elim['area']
ww = ww.fillna(method='ffill')

#calculate weighted mean, min and max of predicted yield (1) including postive change rates
wmean_y1_weigh = round(np.average(w_yield_y1, weights=LoI_welim['area']), 2)  # 2252.36kg/ha
wmax_y1 = w_yield_y1.max()  # 5525.45 kg/ha
wmin_y1 = w_yield_y1.min()  # 588.96 kg/ha
#(2) excluding postive change rates
wmean_y1_0 = round(np.average(w_y1_y0, weights=LoI_welim['area']),2) #2031.69kg/ha
wmax_y10 = w_y1_y0.max()  # 5121.93kg/ha
wmin_y10 = w_y1_y0.min()  # 74.1kg/ha

#change rate
wmean_y1c_weigh = round(np.average(w_y1_change, weights=ww), 2) #-0.28 (~-28%)
wmax_y1c = w_y1_change.max()  # +40.39 (~+4000%)
wmin_y1c = w_y1_change.min()  # -0.9330 (~-93%)

### Phase 2 ###

#select the rows from LoI_welim which contain the independent variables for phase 2
LoI_w_phase2 = LoI_welim.iloc[:, [13, 14, 15, 16, 19, 23, 24, 26]]
#reorder the columns according to the order in dw0_elim
LoI_w_phase2 = LoI_w_phase2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2',
                           'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_w_phase2 = LoI_w_phase2.rename(columns={'p_fert_y2': "p_fertilizer", 'man_fert': "n_total",
                                          'pest_y2': "pesticides_H", 'mechanized_y2': "mechanized",
                                          'irr_LoI': "irrigation_tot"}, errors="raise")

#predict the yield for phase 2 using the gamma GLM
w_yield_y2 = w_fit_elimg.predict(LoI_w_phase2)
#calculate the change from actual yield to the predicted yield
w_y2_change = ((w_yield_y2-wheat_kgha)/wheat_kgha).dropna()
#calculate the number of cells with a postivie change rate
s2 = w_y2_change.loc[w_y2_change > 0]

#create a new variable with all positive change rates set to 0 for both phases
w_c0 = pd.concat([w_y1_change, w_y2_change], axis=1)
w_c0 = w_c0.rename(columns={0: "w_y1_c0", 1: "w_y2_c0"}, errors="raise")
w_c0.loc[w_c0['w_y1_c0'] > 0, 'w_y1_c0'] = 0
w_c0.loc[w_c0['w_y2_c0'] > 0, 'w_y2_c0'] = 0

#create a new variable with the yields for positive change rates set to orginial yields
w02 = w_y2_change.loc[w_y2_change > 0]
w_y2_0 = LoI_welim['Y']
w_y2_0 = w_y2_0[w02.index]
w022 = w_y2_change.loc[w_y2_change <= 0]
w_y2_1 = w_yield_y2[w022.index]
w_y2_y0 = w_y2_0.append(w_y2_1)

# calculate statistics for yield and change rate

#calculate weighted mean, min and max of predicted yield (1) including postive change rates
wmean_y2_weigh = round(np.average(w_yield_y2, weights=LoI_welim['area']), 2)  # 1799.3kg/ha
wmax_y2 = w_yield_y2.max()  # 4310.70g/ha
wmin_y2 = w_yield_y2.min()  # 579.24kg/ha
#(2) excluding postive change rates
wmean_y2_0 = round(np.average(w_y2_y0, weights=LoI_welim['area']),2) #1640.8 kg/ha
wmax_y20 = w_y2_y0.max()  # 4310.70kg/ha
wmin_y20 = w_y2_y0.min()  # 74.1kg/ha

#calculate weighted mean, min and max of predicted change rate (1) including postive change rates
wmean_y2c_weigh = round(np.average(w_y2_change, weights=ww), 2)  #-0.43 (~-43%)
wmax_y2c = w_y2_change.max()  # 34.56 (~+3456%)
wmin_y2c = w_y2_change.min()  # -0.9394 (~-94%)
#(2) excluding postive change rates
wmean_y2c0_weigh = round(np.average(w_c0['w_y2_c0'], weights=ww), 2)  # -0.46
wmean_y1c0_weigh = round(np.average(w_c0['w_y1_c0'], weights=ww), 2)  # -0.35

'''
Statistics to compare current SPAM2010 yield with (1) current fitted values,
(2) phase 1 and (3) phase 2 predictions
'''

## calculate statistics for current yield ##

#SPAM2010 yield: weighted mean, max, min, total yearly production
dw0_mean = round(np.average(dw0_elim['Y'], weights=dw0_elim['area']),2)
dw0_max = dw0_elim['Y'].max()
dw0_min = dw0_elim['Y'].min()
dw0_prod = (dw0_elim['Y'] * dw0_elim['area']).sum()
#fitted values for current yield based on Gamma GLM: wieghted mean, max and min
w_fit_mean = round(np.average(w_fit_elimg.fittedvalues, weights=dw0_elim['area']),2)
w_fit_max = w_fit_elimg.fittedvalues.max()
w_fit_min = w_fit_elimg.fittedvalues.min()
w_fit_prod = (w_fit_elimg.fittedvalues * dw0_elim['area']).sum()

## calculate statistics for both phases ##

#phase 1: calculate the percentage of current yield/production will be achieved
#in phase 1 as predicted by the GLM, calculate total production in phase 1
#(1) including positive change rates and 
w_y1_per = wmean_y1_weigh/dw0_mean  # ~72.2% of current average yield
w_y1_prod = (w_yield_y1 * LoI_welim['area']).sum()
#(2) with positive change rates set to 0
w_y10_prod = (w_y1_y0 * LoI_welim['area']).sum()
w_y10_per = w_y10_prod/dw0_prod

#phase 2: calculate the percentage of current yield/production will be achieved
#in phase 2 as predicted by the GLM, calculate total production in phase 1
#(1) including positive change rates and 
w_y2_per = wmean_y2_weigh/dw0_mean  # 57.4% of current average yield
w_y2_prod = (w_yield_y2 * LoI_welim['area']).sum()
#(2) with positive change rates set to 0
w_y20_prod = (w_y2_y0 * LoI_welim['area']).sum()
w_y20_per = w_y20_prod/dw0_prod

#print the relevant statistics of SPAM2010, fitted values, phase 1 and phase 2
#predictions in order to compare them
#1st column: weighted mean
#2nd row: total crop production in one year
#3rd row: maximum values
#4th row: minimum values
#last two rows comprise statistics for phase 1 and 2 (1) including positive change rates
#and (2) having them set to 0
#5th row: percentage of current yield/production achieved in each phase
#6th row: mean yield change rate for each phase
print(dw0_mean, w_fit_mean, wmean_y1_weigh, wmean_y2_weigh,
      dw0_prod, w_fit_prod, w_y1_prod, w_y2_prod, 
      dw0_max, w_fit_max, wmax_y1, wmax_y2, 
      dw0_min, w_fit_min, wmin_y1, wmin_y2,
      w_y1_per, w_y2_per, w_y10_per, w_y20_per,
      wmean_y1c_weigh, wmean_y2c_weigh, wmean_y1c0_weigh, wmean_y2c0_weigh)


'''
save the predicted yields and the yield change rates for each phase
'''

#combine yields and change rates of both phases with the latitude and longitude values
LoI_wheat = pd.concat([wheat_yield['lats'], wheat_yield['lons'], w_yield_y1,
                       w_y1_change, w_yield_y2, w_y2_change, w_c0], axis='columns')
LoI_wheat = LoI_wheat.rename(columns={0: "w_yield_y1", 1: "w_y1_change",
                                      2: "w_yield_y2", 3: "w_y2_change"}, errors="raise")
# save the dataframe in a csv
LoI_wheat.to_csv(params.geopandasDataDir + "LoIWheatYieldHighRes.csv")

#save the yield for phase 1 and 2 and the change rate for phase 1 and 2 with and without positive rates
#as ASCII files
utilities.create5minASCIIneg(LoI_wheat, 'w_y1_change', params.asciiDir+'LoIWheatYieldChange_y1')
utilities.create5minASCIIneg(LoI_wheat, 'w_yield_y1', params.asciiDir+'LoIWheatYield_y1')
utilities.create5minASCIIneg(LoI_wheat, 'w_y2_change', params.asciiDir+'LoIWheatYieldChange_y2')
utilities.create5minASCIIneg(LoI_wheat, 'w_yield_y2', params.asciiDir+'LoIWheatYield_y2')
utilities.create5minASCIIneg(LoI_wheat, 'w_y1_c0', params.asciiDir+'LoIWheatYieldChange_0y1')
utilities.create5minASCIIneg(LoI_wheat, 'w_y2_c0', params.asciiDir+'LoIWheatYieldChange_0y2')
