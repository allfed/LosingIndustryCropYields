'''

File containing the code to prepare the input data and perform a multiple regression
on yield for rice at 5 arcmin resolution


Jessica Mörsdorf
jessica@allfed.info
jessica.m.moersdorf@umwelt.uni-giessen.de

'''

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utilities import params  # get file location and varname parameters for data import
from src.utilities.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import src.utilities.utilities as utilities

from scipy import stats
import matplotlib
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.metrics import d2_tweedie_score
from sklearn.metrics import mean_tweedie_deviance

params.importAll()


######################################################################
############# Data import and Pre-Analysis Processing#################
######################################################################


'''
Import yield data, extract zeros and plot the data
'''

#import yield data for rice

rice_yield=pd.read_csv(params.geopandasDataDir + 'RICECropYieldHighRes.csv')

#select all rows from rice_yield for which the column growArea has a value greater than zero
rice_nozero=rice_yield.loc[rice_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
rice_kgha=rice_nozero['yield_kgPerHa']

round(np.average(rice_kgha, weights=rice_nozero['growArea']),2)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot rice yield distribution in a histogram
plt.hist(rice_kgha, bins=50)
plt.title('rice yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')


'''
Fitting of distributions to the data and comparing the fit
'''

#calculate loglik, AIC & BIC for each distribution
#st = stat_ut.stat_overview(dist_listr, pdf_listr, param_dictr)
#
#       Distribution  loglikelihood           AIC           BIC
#7  normal on log  -3.743533e+05  7.487227e+05  7.488078e+05
#6  Inverse Gamma  -2.782122e+06  5.564261e+06  5.564346e+06
#4     halfnormal  -2.785834e+06  5.571683e+06  5.571768e+06
#1    exponential  -2.800599e+06  5.601214e+06  5.601299e+06
#3         normal  -2.843839e+06  5.687694e+06  5.687779e+06
#5          Gamma  -3.058454e+06  6.116923e+06  6.117008e+06
#0        lognorm  -3.407763e+06  6.815542e+06  6.815627e+06
#2        weibull  -3.503202e+06  7.006420e+06  7.006505e+06
#best fit so far: normal on log values by far, then Inverse Gamma on non-log


'''
Import factor datasets and extract zeros,
Harmonize units and correct irrigation fraction
'''
r_pesticides=pd.read_csv(params.geopandasDataDir + 'RicePesticidesHighRes.csv')
fertilizer=pd.read_csv(params.geopandasDataDir + 'FertilizerHighRes.csv') #kg/m²
fertilizer_man=pd.read_csv(params.geopandasDataDir + 'FertilizerManureHighRes.csv') #kg/km²
irr_t=pd.read_csv(params.geopandasDataDir + 'FracIrrigationAreaHighRes.csv')
crop = pd.read_csv(params.geopandasDataDir + 'FracCropAreaHighRes.csv')
irr_rel=pd.read_csv(params.geopandasDataDir + 'FracReliantHighRes.csv')
tillage=pd.read_csv(params.geopandasDataDir + 'TillageAllCropsHighRes.csv')
aez=pd.read_csv(params.geopandasDataDir + 'AEZHighRes.csv')

#fraction of irrigation total is of total cell area so it has to be divided by the
#fraction of crop area in a cell and set all values >1 to 1
irr_tot = irr_t['fraction']/crop['fraction']
irr_tot.loc[irr_tot > 1] = 1
#dividing by 0 leaves a NaN value, have to be set back to 0
irr_tot.loc[irr_tot.isna()] = 0

#fertilizer is in kg/m² and fertilizer_man is in kg/km² while yield and pesticides are in kg/ha
#all continuous variables are transformed to kg/ha
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
Loading variables into a combined dataframe and preparing the input
data for analysis by filling/eliminating missing data points, deleting
outliers and combining levels of categorical factors
'''

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
#select only the rows where the area of the cropland is larger than 100 ha
dr0_raw=drice_raw.loc[drice_raw['area'] > 100]

dr0_raw['pesticides_H'] = dr0_raw['pesticides_H'].replace(np.nan, -9)
dr0_raw['irrigation_rel'] = dr0_raw['irrigation_rel'].replace(np.nan, -9)

#replace 0s in the moisture, temperature and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
dr0_raw['thz_class'] = dr0_raw['thz_class'].replace(0,np.nan)
dr0_raw['mst_class'] = dr0_raw['mst_class'].replace(0,np.nan)
dr0_raw['soil_class'] = dr0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 8,9 & 10 with 7 in the temperature class to combine all three classes
#into one Temp,cool-Arctic class
#repalce 2 with 1 and 7 with 6 in the moisture class to compile them into one class each
dr0_raw['thz_class'] = dr0_raw['thz_class'].replace([8,9,10],7)
dr0_raw['mst_class'] = dr0_raw['mst_class'].replace(2,1)
dr0_raw['mst_class'] = dr0_raw['mst_class'].replace(7,6)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the preceding cell)
dr0_raw = dr0_raw.fillna(method='ffill')

#Eliminate the rows without data:
dr0_elim = dr0_raw.loc[dr0_raw['pesticides_H'] > -9]
dr0_elim = dr0_elim.loc[dr0_elim['mechanized'] > -9] 

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
#because there are only few left
dr0_elim.loc[dr0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan
dr0_elim.loc[dr0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
dr0_elim = dr0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
dr0_elim.loc[dr0_elim['n_total'] < 0, 'n_total'] = dr0_elim['n_fertilizer'] + dr0_elim['n_manure']

#calculate the 25th, 50th, 75th, 85th, 95th, 99th and 99.9th percentile
dr0_qt = dr0_elim.quantile([ .25, .5, .75, .85, .95, .99, .999])
dr0_qt.reset_index(inplace=True, drop=True)

#Values above the 99.9th percentile are considered unreasonable outliers
#Calculate number and statistic properties of the outliers
Y_out = dr0_elim.loc[dr0_elim['Y'] > dr0_qt.iloc[6,3]]
nf_out = dr0_elim.loc[dr0_elim['n_fertilizer'] > dr0_qt.iloc[6,4]]
pf_out = dr0_elim.loc[dr0_elim['p_fertilizer'] > dr0_qt.iloc[6,5]] 
nm_out = dr0_elim.loc[dr0_elim['n_manure'] > dr0_qt.iloc[5,6]] 
nt_out = dr0_elim.loc[dr0_elim['n_total'] > dr0_qt.iloc[6,8]] 
P_out = dr0_elim.loc[dr0_elim['pesticides_H'] > dr0_qt.iloc[6,9]]
r_out = pd.concat([Y_out['Y'], nf_out['n_fertilizer'], pf_out['p_fertilizer'],
                 nm_out['n_manure'], nt_out['n_total'], P_out['pesticides_H']], axis=1)
r_out.max()
r_out.min()
r_out.mean()

#Eliminate all points above the 99.9th percentile
dr0_elim = dr0_elim.loc[dr0_elim['Y'] < dr0_qt.iloc[6,3]]
dr0_elim = dr0_elim.loc[dr0_elim['n_fertilizer'] < dr0_qt.iloc[6,4]]
dr0_elim = dr0_elim.loc[dr0_elim['p_fertilizer'] < dr0_qt.iloc[6,5]]
dr0_elim = dr0_elim.loc[dr0_elim['n_manure'] < dr0_qt.iloc[5,6]]
dr0_elim = dr0_elim.loc[dr0_elim['n_total'] < dr0_qt.iloc[6,8]]
dr0_elim = dr0_elim.loc[dr0_elim['pesticides_H'] < dr0_qt.iloc[6,9]]

'''
Dummy-code the categorical variables to be able to assess multicollinearity
'''

#mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########
dur_mst_elim = pd.get_dummies(dr0_elim['mst_class'])
dur_thz_elim = pd.get_dummies(dr0_elim['thz_class'])
dur_soil_elim = pd.get_dummies(dr0_elim['soil_class'])
#####Rename Columns##########
dur_mst_elim = dur_mst_elim.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
dur_thz_elim = dur_thz_elim.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
dur_soil_elim = dur_soil_elim.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
drice_d_elim = pd.concat([dr0_elim, dur_mst_elim, dur_thz_elim, dur_soil_elim], axis='columns')
#drop one column of each dummy (this value will be encoded by 0 in all columns)
drice_dur_elim = drice_d_elim.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')

'''
Split the data into a validation and a calibration dataset
'''

#select a random sample of 20% from the dataset to set aside for later validation
#randor_state argument ensures that the same sample is returned each time the code is run
drice_val_elim = drice_dur_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for calibrating the model
drice_fit_elim = drice_dur_elim.drop(drice_val_elim.index)

'''
Check for multicollinearity by calculating the two-way correlations and the VIF
'''

#extract lat, lon, area, yield, individual n columns, original climate class columns and irrigation for the LoI scenario
#from the fit dataset to test the correlations among the
#independent variables
drice_cor_elim = drice_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')

#### Correlations ###

#calculate spearman (rank transformed) correlation coeficcients between the 
#independent variables and save the values in a dataframe
sp_r = drice_cor_elim.corr(method='spearman')

### Variance inflation factor ###

Xr = add_constant(drice_cor_elim)
pd.Series([variance_inflation_factor(Xr.values, i) 
               for i in range(Xr.shape[1])], 
              index=Xr.columns)

#const                74.703572
#p_fertilizer          8.900688
#n_total               8.657282
#pesticides_H          2.818537
#mechanized            1.362047
#irrigation_tot        1.953442
#LGP<120days           1.379230
#120-180days           1.636072
#180-225days           1.502411
#225-270days           1.328974
#Trop_low             12.422695
#Trop_high             1.756088
#Sub-trop_warm         8.000736
#Sub-trop_mod_cool     5.289685
#Sub-trop_cool         2.420536
#Temp_mod              3.960040
#S1_very_steep         1.276521
#S2_hydro_soil         1.190554
#S3_no-slight_lim      1.865182
#S4_moderate_lim       2.469434
#S5_severe_lim         1.536933


#######################################################################
########### Regression Calibration, Validation and Residuals###########
#######################################################################

'''
Calibrate the Regression model and calculate fit statistics
'''

#determine model with a gamma distribution
r_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=drice_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
r_mod_elim0 = smf.glm(formula='Y ~ 1', data=drice_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))

#Fit models
r_fit_elimg = r_mod_elimg.fit()
r_fit_elim0 = r_mod_elim0.fit()

#print results
print(r_fit_elimg.summary())
print(r_fit_elim0.summary())

#calculate the odds ratios on the response scale
np.exp(r_fit_elimg.params)

### Fit statistics ###

#calculate McFadden's roh² and the Root Mean Gamma Deviance (RMGD)
d2_tweedie_score(drice_fit_elim['Y'], r_fit_elimg.fittedvalues, power=2) #0.41
np.sqrt(mean_tweedie_deviance(drice_fit_elim['Y'], r_fit_elimg.fittedvalues, power=2)) #0.4954

#calculate AIC and BIC for Gamma
r_aic = r_fit_elimg.aic 
r_bic = r_fit_elimg.bic_llf
#LogLik: -670710; AIC: 1341464; BIC: 1341658


'''
Validate the model against the validation dataset
'''

#select the independent variables from the validation dataset
r_val_elim = drice_val_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#let the model predict yield values for the validation data
r_pred_elimg = r_fit_elimg.predict(r_val_elim)

#calculate McFadden's roh² and the RMGD scores
d2_tweedie_score(drice_val_elim['Y'], r_pred_elimg, power=2)#0.4114
np.sqrt(mean_tweedie_deviance(drice_val_elim['Y'], r_pred_elimg, power=2)) #0.4913

'''
Plot the Residuals for the model
'''
### Extract necessary measures ###

#select the independent variables from the fit dataset
r_fit_elim = drice_fit_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#get the influence of the GLM model
r_stat_elimg = r_fit_elimg.get_influence()

#store cook's distance in a variable
r_elimg_cook = pd.Series(r_stat_elimg.cooks_distance[0]).transpose()
r_elimg_cook = r_elimg_cook.rename("Cooks_d", errors="raise")

#store the actual yield, the fitted values on response and link scale, 
#the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
#the absolute value of the resp and the sqrt of the stud residuals in a dataframe
#reset the index but keep the old one as a column in order to combine the dataframe
#with Cook's distance
r_data_infl = { 'Yield': drice_fit_elim['Y'],
                'GLM_fitted': r_fit_elimg.fittedvalues,
               'Fitted_link': r_fit_elimg.predict(r_fit_elim, linear=True),
               'resid_pear': r_fit_elimg.resid_pearson, 
               'resid_stud' : r_stat_elimg.resid_studentized,
               'resid_resp_abs' : np.abs(r_fit_elimg.resid_response),
               'resid_stud_sqrt' : np.sqrt(np.abs(r_stat_elimg.resid_studentized)),
               'hat_matrix':r_stat_elimg.hat_matrix_diag}
r_elimg_infl = pd.DataFrame(data=r_data_infl).reset_index()
r_elimg_infl = pd.concat([r_elimg_infl, r_elimg_cook], axis='columns')

#take a sample of the influence dataframe to plot the lowess line
r_elimg_infl_sample = r_elimg_infl.sample(frac=0.1, random_state=2705)

### Studentized residuals vs. fitted values on link scale ###

#set plot characteristics
plot_rs = plt.figure(4)
plot_rs.set_figheight(8)
plot_rs.set_figwidth(12)

#Draw a scatterplot of studentized residuals vs. fitted values on the link scale
#lowess=True draws a fitted line which shows if the relationship is linear
plot_rs.axes[0] = sb.regplot('Fitted_link', 'resid_stud', data=r_elimg_infl_sample, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
#plt.scatter('Fitted_link', 'resid_stud', data=r_elimg_infl)

#plot labels
plot_rs.axes[0].set_title('Studentized Residuals vs Fitted on link scale')
plot_rs.axes[0].set_xlabel('Fitted values on link scale')
plot_rs.axes[0].set_ylabel('Studentized Residuals')

### Response residuals vs. fitted values on the response scale ###

#set plot characteristics
plot_rr = plt.figure(4)
plot_rr.set_figheight(8)
plot_rr.set_figwidth(12)

#Draw a scatterplot of response residuals vs. fitted values on the response scale
plot_rr.axes[0] = sb.residplot('GLM_fitted', 'Yield', data=r_elimg_infl, 
                          #lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#plot labels
plot_rr.axes[0].set_title('Residuals vs Fitted')
plot_rr.axes[0].set_xlabel('Fitted values')
plot_rr.axes[0].set_ylabel('Residuals')

#annotations of the three largest residuals
abs_resid = r_elimg_infl['resid_resp_abs'].sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_rr.axes[0].annotate(i, 
                               xy=(r_elimg_infl['GLM_fitted'][i], 
                                   r_elimg_infl['resid_resp_abs'][i]))

### QQ-Plot for the studentized residuals ###

#Specifications of the QQ Plot
QQ = ProbPlot(r_elimg_infl['resid_stud'], dist=stats.gamma, fit=True)
plot_rq = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

#set plot characteristics
plot_rq.set_figheight(8)
plot_rq.set_figwidth(12)

#plot labels
plot_rq.axes[0].set_title('Normal Q-Q')
plot_rq.axes[0].set_xlabel('Theoretical Quantiles')
plot_rq.axes[0].set_ylabel('Standardized Residuals');

#annotations of the three largest residuals
abs_norm_resid = np.flip(np.argsort(np.abs(r_elimg_infl['resid_stud'])), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_rq.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   r_elimg_infl['resid_stud'][i]));

### Cook's distance plots ###

##Cook's distance vs. no of observation##

#sort cook's distance value to get the value for the largest distance####
r_cook_sort = r_elimg_cook.sort_values(ascending=False)
#select all Cook's distance values which are greater than 4/n (n=number of datapoints)
r_cook_infl = r_elimg_cook.loc[r_elimg_cook > (4/(75194-21))].sort_values(ascending=False)

#barplot for values with the strongest influence (=largest Cook's distance)
#because running the function on all values takes a little longer
plt.bar(r_cook_infl.index, r_cook_infl)
plt.ylim(0, 0.01)

#plots for the ones greater than 4/n and all distance values
plt.scatter(r_cook_infl.index, r_cook_infl)
plt.scatter(r_elimg_cook.index, r_elimg_cook)
plt.ylim(0, 0.01)

##Studentized Residuals vs. Leverage w. Cook's distance line##

#set plot characteristics
plot_rc = plt.figure(4)
plot_rc.set_figheight(8)
plot_rc.set_figwidth(12)

#Draw the scatterplott of the Studentized residuals vs. leverage
plt.scatter(r_elimg_infl['hat_matrix'], r_elimg_infl['resid_stud'], alpha=0.5)
sb.regplot(r_elimg_infl['hat_matrix'], r_elimg_infl['resid_stud'], 
            scatter=False, 
            ci=False, 
            #lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#plot boundaries and labels
plot_rc.axes[0].set_xlim(0, 0.004)
plot_rc.axes[0].set_ylim(-3, 21)
plot_rc.axes[0].set_title('Residuals vs Leverage')
plot_rc.axes[0].set_xlabel('Leverage')
plot_rc.axes[0].set_ylabel('Standardized Residuals')

#annotate the three points with the largest Cooks distance value
leverage_top_3 = np.flip(np.argsort(r_elimg_infl["Cooks_d"]), 0)[:3]

for i in leverage_top_3:
    plot_rc.axes[0].annotate(i, 
                               xy=(r_elimg_infl['hat_matrix'][i], 
                                   r_elimg_infl['resid_stud'][i]))


###########################################################################
################ Loss of Industry Modelling ###############################
###########################################################################

'''
Prepare and modify datasets according to the assumptions of the LoI scenario
'''

#take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
#and available manure correctly
LoI_relim = dr0_raw

LoI_relim['mechanized'] = LoI_relim['mechanized'].replace(-9,np.nan)
LoI_relim['pesticides_H'] = LoI_relim['pesticides_H'].replace(-9,np.nan)

### Mechanised ###

#set mechanization to 0 in phase 2; due to the estimated stock in  fuel the variable remains 
#unchanged in phase 1
LoI_relim['mechanized_y2'] = LoI_relim['mechanized'].replace(1,0)

### N fertilizer ###

#drop all cells where mechanized or pesticiedes AND n_fertilizer are no data values
rn_drop= LoI_relim[((LoI_relim['mechanized'].isna())|(LoI_relim['pesticides_H'].isna()))
                & (LoI_relim['n_fertilizer']<0)].index
LoI_relim_pn = LoI_relim.drop(rn_drop)

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_relim_pn.loc[LoI_relim_pn['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan
LoI_relim_pn.loc[LoI_relim_pn['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
LoI_relim_pn[['n_fertilizer','p_fertilizer']] = LoI_relim_pn[['n_fertilizer','p_fertilizer']].fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
LoI_relim_pn.loc[LoI_relim_pn['n_total'] < 0, 'n_total'] = LoI_relim_pn['n_fertilizer'] + LoI_relim_pn['n_manure']

#drop the outliers (99.9th percentile) in the n and p fertilizer columns
LoI_relim_pn = LoI_relim_pn.loc[LoI_relim_pn['n_fertilizer'] < dr0_qt.iloc[6,4]]
LoI_relim_pn = LoI_relim_pn.loc[LoI_relim_pn['p_fertilizer'] < dr0_qt.iloc[6,5]]

#in phase 1, there will probably be a slight surplus of N (production>application)
#the surplus is assumed to be the new total

#calculate kg N applied per cell
LoI_relim_pn['n_kg'] = LoI_relim_pn['n_fertilizer']*LoI_relim_pn['area']
#calculate the fraction of the total N applied to rice fields for each cell
LoI_relim_pn['n_ffrac'] = LoI_relim_pn['n_kg']/(LoI_relim_pn['n_kg'].sum())

#calculate the fraction of total N applied to rice fields on the total N applied
#divide total of rice N by 1000000 to get from kg to thousand t
r_nfert_frac = (LoI_relim_pn['n_kg'].sum())/1000000/118763
#calculate the new total for N rice in phase one based on the N total surplus
r_ntot_new = r_nfert_frac * 14477 * 1000000

#calculate the new value of N application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_relim_pn['n_fert_y1'] = (r_ntot_new * LoI_relim_pn['n_ffrac']) / LoI_relim_pn['area']

#in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_relim_pn['n_fert_y2'] = 0

### P Fertilizer ###

#in phase 1, there will probably be a slight surplus of P (production>application)
#calculate kg p applied per cell
LoI_relim_pn['p_kg'] = LoI_relim_pn['p_fertilizer']*LoI_relim_pn['area']
#calculate the fraction of the total N applied to rice fields for each cell
LoI_relim_pn['p_ffrac'] = LoI_relim_pn['p_kg']/(LoI_relim_pn['p_kg'].sum())

#calculate the fraction of total P applied to rice fields on the total P applied to cropland
#divide total of rice P by 1000000 to get from kg to thousand t
r_pfert_frac = (LoI_relim_pn['p_kg'].sum())/1000000/45858
#calculate the new total for P rice in phase one based on the P total surplus
r_ptot_new = r_pfert_frac * 4142 * 1000000

#calculate the new value of P application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_relim_pn['p_fert_y1'] = (r_ptot_new * LoI_relim_pn['p_ffrac']) / LoI_relim_pn['area']

#in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_relim_pn['p_fert_y2'] = 0

### N Manure ###

#drop the rows containing outliers (99th percentile) in the manure column
LoI_relim_man = LoI_relim.loc[LoI_relim['n_manure'] < dr0_qt.iloc[5,6]]

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_relim_man['labor'] = LoI_relim_man['area']/5 #current value (7.4) is taken from Cole et al. (2016)
#due to information from a farmer, the value is set at 5

#calculate mean excretion rate of each cow in one year: cattle supplied ~ 43.7% of 131000 thousand t
#manure production in 2014, there were ~ 1.439.413.930(FAOSTAT) 
#heads of cattle in 2014
cow_excr = 131000000000*0.437/1439413930

#calculate the new value of man application rate in kg per ha per cell, according
#to the available cows in each cell due to labor demand
LoI_relim_man['man_fert'] = (cow_excr * LoI_relim_man['labor']) / LoI_relim_man['area']

#that leads the application rate being the same in every cell because everybody has the same number of cows per ha
#it's assumed to be the same for both phases

### N total ###

#in phase 1, the total N available is the sum of available fertilizer and manure
LoI_relim['N_toty1'] = LoI_relim_pn['n_fert_y1'] + LoI_relim_man['man_fert']

#in phase 2 there is no more artificial fertilizer, so N total is equal to man_fert

### Pesticides ###

#drop the cells containing NaN values and outliers
LoI_relimp = LoI_relim.loc[LoI_relim['pesticides_H'].notna()]
LoI_relimp = LoI_relimp.loc[LoI_relimp['pesticides_H'] < dr0_qt.iloc[6,9]]

#in phase 1, there will probably be a slight surplus of Pesticides (production>application)

#calculate kg p applied per cell
LoI_relimp['pest_kg'] = LoI_relimp['pesticides_H']*LoI_relimp['area']
#calculate the fraction of the total N applied to rice fields for each cell
LoI_relimp['pest_frac'] = LoI_relimp['pest_kg']/(LoI_relimp['pest_kg'].sum())

#calculate the fraction of total pesticides applied to rice fields on the total pesticides applied to cropland
#divide total of rice pesticides by 1000 to get from kg to t
r_pest_frac = (LoI_relimp['pest_kg'].sum())/1000/4190985

#due to missing reasonable data on the pesticide surplus, it is assumed that the
#surplus is in the same range as for P and N fertilizer
#the mean of N and P fertilizer surplus is calculated
frac_pest = ((14477/118763) + (4142/45858))/2
#calculate the new total for pesticides rice in phase one based on the pesticides total surplus
r_pestot_new = r_pest_frac * (4190985 * frac_pest) * 1000

#calculate the new value of pesticides application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_relimp['pest_y1'] = (r_pestot_new * LoI_relimp['pest_frac']) / LoI_relimp['area']

#in phase 2 no industrially produced pesticides will be available anymore: set to 0
LoI_relimp['pest_y2'] = 0

### Irrigation ###

#in LoI it is assumed that only irrigation which is not reliant on electricity
#can still be maintained
#calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
#(fraction of irrigated cropland reliant on electricity)
LoI_relim['irr_LoI'] = LoI_relim['irrigation_tot'] * (1- LoI_relim['irrigation_rel'])

### Combine the different dataframes and drop rows with missing values ###

LoI_relim = pd.concat([LoI_relim, LoI_relim_pn['n_fert_y1'], LoI_relim_pn['n_fert_y2'],
                       LoI_relim_pn['p_fert_y1'], LoI_relim_pn['p_fert_y2'],
                       LoI_relim_man['man_fert'], LoI_relimp['pest_y1'], 
                       LoI_relimp['pest_y2']], axis='columns')

#Eliminate the rows without data:
LoI_relim = LoI_relim.dropna()

#Eliminating all points above the 99.9th percentile
LoI_relim = LoI_relim.loc[LoI_relim['Y'] < dr0_qt.iloc[6,3]]
LoI_relim = LoI_relim.loc[LoI_relim['n_total'] < dr0_qt.iloc[6,8]]


'''
Prediction of LoI yields and yield change rates in phase 1 and 2
'''
### Phase 1 ###

#select the rows from LoI_relim which contain the independent variables for phase 1
LoI_r_phase1 = LoI_relim.iloc[:,[10,13,14,15,17,19,22,25]]
#reorder the columns according to the order in dr0_elim
LoI_r_phase1 = LoI_r_phase1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_r_phase1 = LoI_r_phase1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                      'pest_y1':"pesticides_H",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")

#predict the yield for phase 1 using the gamma GLM
r_yield_y1 = r_fit_elimg.predict(LoI_r_phase1)
#calculate the change rate from actual yield to the predicted yield
r_y1_change = ((r_yield_y1-rice_kgha)/rice_kgha).dropna()
#calculate the number of cells with a postivie change rate
s1 = r_y1_change.loc[r_y1_change>0]

#create a new variable with the yields for positive change rates set to orginial yields
r01 = r_y1_change.loc[r_y1_change > 0]
r_y1_0 = LoI_relim['Y']
r_y1_0 = r_y1_0[r01.index]
r011 = r_y1_change.loc[r_y1_change <= 0]
r_y1_1 = r_yield_y1[r011.index]
r_y1_y0 = r_y1_0.append(r_y1_1)

#calculate statistics for yield and change rate

#calculate weights for mean change rate calculation dependent on current yield
#and current rice area in a cell
rw=LoI_relim['Y']*dr0_elim['area']
rw = rw.fillna(method='ffill')

#calculate weighted mean, min and max of predicted yield (1) including postive change rates
rmean_y1_weigh = round(np.average(r_yield_y1, weights=LoI_relim['area']),2) #3148.82kg/ha
rmax_y1 = r_yield_y1.max() #7765.09 kg/ha
rmin_y1 = r_yield_y1.min() #1361.24 kg/ha
#(2) excluding postive change rates
rmean_y1_0 = round(np.average(r_y1_y0, weights=LoI_relim['area']),2) #2871.9kg/ha
rmax_y10 = r_y1_y0.max()  # 7506.11kg/ha
rmin_y10 = r_y1_y0.min()  # 111.2kg/ha

#change rate
rmean_y1c_weigh = round(np.average(r_y1_change, weights=rw),2) #-0.28 (~-28%)
rmax_y1c = r_y1_change.max() # +24.92 (~+2500%)
rmin_y1c = r_y1_change.min() #-0.8567 (~-86%)

### Phase 2 ###

#select the rows from LoI_relim which contain the independent variables for phase 2
LoI_r_phase2 = LoI_relim.iloc[:,[13,14,15,16,19,23,24,26]]
#reorder the columns according to the order in dr0_elim
LoI_r_phase2 = LoI_r_phase2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_r_phase2 = LoI_r_phase2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                      'pest_y2':"pesticides_H",'mechanized_y2':"mechanized",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")

#predict the yield for phase 2 using the gamma GLM
r_yield_y2 = r_fit_elimg.predict(LoI_r_phase2)
#calculate the change from actual yield to the predicted yield
r_y2_change = ((r_yield_y2-rice_kgha)/rice_kgha).dropna()
#calculate the number of cells with a postivie change rate
s2 = r_y2_change.loc[r_y2_change>0]

#create a new variable with all positive change rates set to 0 for both phases
r_c0 = pd.concat([r_y1_change, r_y2_change], axis=1)
r_c0 = r_c0.rename(columns={0:"r_y1_c0", 1:"r_y2_c0"}, errors="raise")
r_c0.loc[r_c0['r_y1_c0'] > 0, 'r_y1_c0'] = 0
r_c0.loc[r_c0['r_y2_c0'] > 0, 'r_y2_c0'] = 0

#create a new variable with the yields for positive change rates set to orginial yields
r02 = r_y2_change.loc[r_y2_change > 0]
r_y2_0 = LoI_relim['Y']
r_y2_0 = r_y2_0[r02.index]
r022 = r_y2_change.loc[r_y2_change <= 0]
r_y2_1 = r_yield_y2[r022.index]
r_y2_y0 = r_y2_0.append(r_y2_1)

#calculate statistics for yield and change rate

#calculate weighted mean, min and max of predicted yield (1) including postive change rates
rmean_y2_weigh = round(np.average(r_yield_y2, weights=LoI_relim['area']),2) #2955.57kg/ha
rmax_y2 = r_yield_y2.max() #6587.54kg/ha
rmin_y2 = r_yield_y2.min() #1355.65kg/ha
#(2) excluding postive change rates
rmean_y2_0 = round(np.average(r_y2_y0, weights=LoI_relim['area']),2) #2725.54 kg/ha
rmax_y20 = r_y2_y0.max()  # 6319.47kg/ha
rmin_y20 = r_y2_y0.min()  # 111.2kg/ha

#calculate weighted mean, min and max of predicted change rate (1) including postive change rates
rmean_y2c_weigh = round(np.average(r_y2_change, weights=rw),2) #-0.29 (~-29%)
rmax_y2c = r_y2_change.max() #+25.03 (~+2500%)
rmin_y2c = r_y2_change.min() #-0.8569 (~-86%)
#(2) excluding postive change rates
rmean_y2c0_weigh = round(np.average(r_c0['r_y2_c0'], weights=rw),2) #-0.37
rmean_y1c0_weigh = round(np.average(r_c0['r_y1_c0'], weights=rw),2) #-0.33

'''
Statistics to compare current SPAM2010 yield with (1) current fitted values,
(2) phase 1 and (3) phase 2 predictions
'''

## calculate statistics for current yield ##

#SPAM2010 yield: weighted mean, max, min, total yearly production
dr0_mean = round(np.average(dr0_elim['Y'], weights=dr0_elim['area']),2)
dr0_max = dr0_elim['Y'].max()
dr0_min = dr0_elim['Y'].min()
dr0_prod = (dr0_elim['Y'] * dr0_elim['area']).sum()
#fitted values for current yield based on Gamma GLM: wieghted mean, max and min
r_fit_mean = round(np.average(r_fit_elimg.fittedvalues, weights=dr0_elim['area']),2)
r_fit_max = r_fit_elimg.fittedvalues.max()
r_fit_min = r_fit_elimg.fittedvalues.min()
r_fit_prod = (r_fit_elimg.fittedvalues * dr0_elim['area']).sum()

## calculate statistics for both phases ##

#phase 1: calculate the percentage of current yield/production will be achieved
#in phase 1 as predicted by the GLM, calculate total production in phase 1
#(1) including positive change rates and 
r_y1_per = rmean_y1_weigh/dr0_mean #~71.7% of current average yield
r_y1_prod = (r_yield_y1 * LoI_relim['area']).sum()
#(2) with positive change rates set to 0
r_y10_prod = (r_y1_y0 * LoI_relim['area']).sum()
r_y10_per = r_y10_prod/dr0_prod

#phase 2: calculate the percentage of current yield/production will be achieved
#in phase 2 as predicted by the GLM, calculate total production in phase 1
#(1) including positive change rates and 
r_y2_per = rmean_y2_weigh/dr0_mean #67.3% of current average yield
r_y2_prod = (r_yield_y2 * LoI_relim['area']).sum()
#(2) with positive change rates set to 0
r_y20_prod = (r_y2_y0 * LoI_relim['area']).sum()
r_y20_per = r_y20_prod/dr0_prod

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
print(dr0_mean, r_fit_mean, rmean_y1_weigh, rmean_y2_weigh,
      dr0_prod, r_fit_prod, r_y1_prod, r_y2_prod, 
      dr0_max, r_fit_max, rmax_y1, rmax_y2, 
      dr0_min, r_fit_min, rmin_y1, rmin_y2,
      r_y1_per, r_y2_per, r_y10_per, r_y20_per,
      rmean_y1c_weigh, rmean_y2c_weigh, rmean_y1c0_weigh, rmean_y2c0_weigh)


'''
save the predicted yields and the yield change rates for each phase
'''

#combine yields and change rates of both phases with the latitude and longitude values
LoI_rice = pd.concat([rice_yield['lats'], rice_yield['lons'], r_yield_y1,
                       r_y1_change, r_yield_y2, r_y2_change, r_c0], axis='columns')
LoI_rice = LoI_rice.rename(columns={0:"r_yield_y1", 1:"r_y1_change", 
                                      2:"r_yield_y2",3:"r_y2_change"}, errors="raise")
#save the dataframe in a csv
LoI_rice.to_csv(params.geopandasDataDir + "LoIRiceYieldHighRes.csv")

#save the yield for phase 1 and 2 and the change rate for phase 1 and 2 with and without positive rates
#as ASCII files
utilities.create5minASCIIneg(LoI_rice,'r_y1_change',params.asciiDir+'LoIRiceYieldChange_y1')
utilities.create5minASCIIneg(LoI_rice,'r_yield_y1',params.asciiDir+'LoIRiceYield_y1')
utilities.create5minASCIIneg(LoI_rice,'r_y2_change',params.asciiDir+'LoIRiceYieldChange_y2')
utilities.create5minASCIIneg(LoI_rice,'r_yield_y2',params.asciiDir+'LoIRiceYield_y2')
utilities.create5minASCIIneg(LoI_rice,'r_y1_c0',params.asciiDir+'LoIRiceYieldChange_0y1')
utilities.create5minASCIIneg(LoI_rice,'r_y2_c0',params.asciiDir+'LoIRiceYieldChange_0y2')
