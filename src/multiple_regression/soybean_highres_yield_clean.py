'''

File containing the code to explore data and perform a multiple regression
on yield for soyb
'''

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
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


######################################################################
############# Data import and Pre-Analysis Processing#################
######################################################################


'''
Import yield data, extract zeros and plot the data
'''

#import yield data for soyb

soyb_yield=pd.read_csv(params.geopandasDataDir + 'SOYBCropYieldHighRes.csv')

#select all rows from soyb_yield for which the column growArea has a value greater than zero
soyb_nozero=soyb_yield.loc[soyb_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
soyb_kgha=soyb_nozero['yield_kgPerHa']

round(np.average(soyb_kgha, weights=soyb_nozero['growArea']),2)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


#plot soyb yield distribution in a histogram
plt.hist(soyb_kgha, bins=50)
plt.title('soyb yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')


'''
Fitting of distributions to the data and comparing the fit
'''


#calculate loglik, AIC & BIC for each distribution
#st = stat_ut.stat_overview(dist_lists, pdf_lists, param_dicts)

#    Distribution  loglikelihood           AIC           BIC
#7  normal on log  -3.748763e+05  7.497686e+05  7.498546e+05
#6  Inverse Gamma  -2.859205e+06  5.718427e+06  5.718513e+06
#5          Gamma  -2.860227e+06  5.720469e+06  5.720555e+06
#3         normal  -2.908569e+06  5.817154e+06  5.817240e+06
#1    exponential  -2.910045e+06  5.820106e+06  5.820192e+06
#0        lognorm  -3.587555e+06  7.175125e+06  7.175211e+06
#2        weibull  -3.694327e+06  7.388671e+06  7.388757e+06
#4     halfnormal           -inf           inf           inf
#Best fit is normal on log by far, then inverse gamma on non-log


'''
Import factor datasets and extract zeros,
Harmonize units and correct irrigation fraction
'''
s_pesticides=pd.read_csv(params.geopandasDataDir + 'SoybeanPesticidesHighRes.csv')
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

datas_raw = {"lat": soyb_yield.loc[:,'lats'],
        "lon": soyb_yield.loc[:,'lons'],
        "area": soyb_yield.loc[:,'growArea'],
        "Y": soyb_yield.loc[:,'yield_kgPerHa'],
        "n_fertilizer": fertilizer.loc[:,'n_kgha'],
        "p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_man_prod" : fertilizer_man.loc[:,'produced_kgha'],
        "n_total" : N_total,
        "pesticides_H": s_pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'is_mech'],
        "irrigation_tot": irr_tot,
        "irrigation_rel": irr_rel.loc[:,'frac_reliant'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
        }

#arrange data_raw in a dataframe
dsoyb_raw = pd.DataFrame(data=datas_raw)

ds0_raw=dsoyb_raw.loc[dsoyb_raw['area'] > 0]
ds0_raw['n_fertilizer'].min()
ds0_raw['n_fertilizer'] = ds0_raw['n_fertilizer'].replace(-99989.9959564209, np.nan)
ds0_raw['n_fertilizer'].isna().sum()
ds0_raw['p_fertilizer'] = ds0_raw['p_fertilizer'].replace(-99989.9959564209, np.nan)
ds0_raw['p_fertilizer'].isna().sum()
ds0_raw['n_total'] = ds0_raw['n_total'].replace(-99989.9959564209, np.nan)
ds0_raw['n_total'].isna().sum()
#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('ds0_raw Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=ds0_raw, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=ds0_raw, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=ds0_raw, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=ds0_raw, x='n_total')
sb.boxplot(ax=axes[1, 1], data=ds0_raw, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=ds0_raw, x='Y')

ax = sb.boxplot(x=ds0_raw["Y"], orient='v')
ax = sb.boxplot(x=ds0_raw["n_fertilizer"])
ax = sb.boxplot(x=ds0_raw["p_fertilizer"])
ax = sb.boxplot(x=ds0_raw["n_manure"])
ax = sb.boxplot(x=ds0_raw["n_total"])
ax = sb.boxplot(x=ds0_raw["pesticides_H"])
ax = sb.boxplot(x=ds0_raw["irrigation_tot"])
ax = sb.boxplot(x=ds0_raw["irrigation_rel"])
ax = sb.boxplot(x="mechanized", y='Y', data=ds0_raw)
ax = sb.boxplot(x="thz_class", y='Y', data=ds0_raw)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=ds0_raw)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=ds0_raw)
plt.ylim(0,20000)


#select only the rows where the area of the cropland is larger than 100 ha
ds0_raw=dsoyb_raw.loc[dsoyb_raw['area'] > 100]

ds0_raw['pesticides_H'] = ds0_raw['pesticides_H'].replace(np.nan, -9)
ds0_raw['irrigation_rel'] = ds0_raw['irrigation_rel'].replace(np.nan, -9)

#replace 0s in the moisture, temperature and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
ds0_raw['thz_class'] = ds0_raw['thz_class'].replace(0,np.nan)
ds0_raw['mst_class'] = ds0_raw['mst_class'].replace(0,np.nan)
ds0_raw['soil_class'] = ds0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 8,9 & 10 with 7 in the temperature class to combine all three classes
#into one Temp,cool-Arctic class
#repalce 2 with 1 and 7 with 6 in the moisture class to compile them into one class each
ds0_raw['thz_class'] = ds0_raw['thz_class'].replace([8,9,10],7)
ds0_raw['mst_class'] = ds0_raw['mst_class'].replace(2,1)
ds0_raw['mst_class'] = ds0_raw['mst_class'].replace(7,6)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
ds0_raw = ds0_raw.fillna(method='ffill')

#Eliminate the rows without data:
ds0_elim = ds0_raw.loc[ds0_raw['pesticides_H'] > -9]
ds0_elim = ds0_elim.loc[ds0_raw['mechanized'] > -9] 

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
#because there are only few left
ds0_elim.loc[ds0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan
ds0_elim.loc[ds0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
ds0_elim = ds0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
ds0_elim.loc[ds0_elim['n_total'] < 0, 'n_total'] = ds0_elim['n_fertilizer'] + ds0_elim['n_manure']

#calculate the 25th, 50th, 75th, 85th, 95th, 99th and 99.9th percentile
ds0_qt = ds0_elim.quantile([ .25, .5, .75, .85, .95, .99, .999])
ds0_qt.reset_index(inplace=True, drop=True)

#Values above the 99.9th percentile are considered unreasonable outliers
#Calculate number and statistic properties of the outliers
Y_out = ds0_elim.loc[ds0_elim['Y'] > ds0_qt.iloc[6,3]]
nf_out = ds0_elim.loc[ds0_elim['n_fertilizer'] > ds0_qt.iloc[6,4]]
pf_out = ds0_elim.loc[ds0_elim['p_fertilizer'] > ds0_qt.iloc[6,5]]
nm_out = ds0_elim.loc[ds0_elim['n_manure'] > ds0_qt.iloc[5,6]] 
nt_out = ds0_elim.loc[ds0_elim['n_total'] > ds0_qt.iloc[6,8]] 
P_out = ds0_elim.loc[ds0_elim['pesticides_H'] > ds0_qt.iloc[6,9]]
s_out = pd.concat([Y_out['Y'], nf_out['n_fertilizer'], pf_out['p_fertilizer'],
                 nm_out['n_manure'], nt_out['n_total'], P_out['pesticides_H']], axis=1)
s_out.max()
s_out.min()
s_out.mean()

#Eliminate all points above the 99.9th percentile
ds0_elim = ds0_elim.loc[ds0_elim['Y'] < ds0_qt.iloc[6,3]]
ds0_elim = ds0_elim.loc[ds0_elim['n_fertilizer'] < ds0_qt.iloc[6,4]]
ds0_elim = ds0_elim.loc[ds0_elim['p_fertilizer'] < ds0_qt.iloc[6,5]]
ds0_elim = ds0_elim.loc[ds0_elim['n_manure'] < ds0_qt.iloc[5,6]]
ds0_elim = ds0_elim.loc[ds0_elim['n_man_prod'] < ds0_qt.iloc[6,7]]
ds0_elim = ds0_elim.loc[ds0_elim['n_total'] < ds0_qt.iloc[6,8]]
ds0_elim = ds0_elim.loc[ds0_elim['pesticides_H'] < ds0_qt.iloc[6,9]]


'''
Dummy-code the categorical variables to be able to assess multicollinearity
'''

#mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########
dus_mst_elim = pd.get_dummies(ds0_elim['mst_class'])
dus_thz_elim = pd.get_dummies(ds0_elim['thz_class'])
dus_soil_elim = pd.get_dummies(ds0_elim['soil_class'])
#####Rename Columns##########
dus_mst_elim = dus_mst_elim.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
dus_thz_elim = dus_thz_elim.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
dus_soil_elim = dus_soil_elim.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
dsoyb_d_elim = pd.concat([ds0_elim, dus_mst_elim, dus_thz_elim, dus_soil_elim], axis='columns')
#drop one column of each dummy (this value will be encoded by 0 in all columns)
dsoyb_dus_elim = dsoyb_d_elim.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')

'''
Split the data into a validation and a calibration dataset
'''

#select a random sample of 20% from the dataset to set aside for later validation
#randor_state argument ensures that the same sample is returned each time the code is run
dsoyb_val_elim = dsoyb_dus_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for calibrating the model
dsoyb_fit_elim = dsoyb_dus_elim.drop(dsoyb_val_elim.index) #RAW

'''
Check for multicollinearity by calculating the two-way correlations and the VIF
'''

#extract lat, lon, area, yield, individual n columns, original climate class columns and irrigation for the LoI scenario
#from the fit dataset to test the correlations among the
#independent variables
dsoyb_cor_elim = dsoyb_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')

#### Correlations ###

#calculate spearman (rank transformed) correlation coeficcients between the 
#independent variables and save the values in a dataframe
sp_s = dsoyb_cor_elim.corr(method='spearman')

### Variance inflation factor ###

Xs = add_constant(dsoyb_cor_elim)
pd.Series([variance_inflation_factor(Xs.values, i) 
               for i in range(Xs.shape[1])], 
              index=Xs.columns)

#const                58.186680
#p_fertilizer          4.455885
#n_total               6.277271
#pesticides_H          2.354473
#mechanized            2.695139
#irrigation_tot        2.208590
#LGP<120days           1.122061
#120-180days           1.772309
#180-225days           1.620584
#225-270days           1.667707
#Trop_low              4.127611
#Trop_high             1.267871
#Sub-trop_warm         2.156898
#Sub-trop_mod_cool     2.469550
#Sub-trop_cool         1.348031
#Temp_mod              2.084801
#S1_very_steep         1.154701
#S2_hydro_soil         1.568389
#S3_no-slight_lim      5.159879
#S4_moderate_lim       4.686159
#S5_severe_lim         1.685200


#######################################################################
########### Regression Calibration, Validation and Residuals###########
#######################################################################

'''
Calibrate the Regression model and calculate fit statistics
'''

#determine model with a gamma distribution
s_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dsoyb_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
s_mod_elim0 = smf.glm(formula='Y ~ 1', data=dsoyb_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))

#Fit models
s_fit_elimg = s_mod_elimg.fit()
s_fit_elim0 = s_mod_elim0.fit()

#print results
print(s_fit_elimg.summary())
print(s_fit_elim0.summary())

#calculate the odds ratios on the response scale
np.exp(s_fit_elimg.params)

### Fit statistics ###

#calculate McFadden's roh² and the Root Mean Gamma Deviance (RMGD)
d2_tweedie_score(dsoyb_fit_elim['Y'], s_fit_elimg.fittedvalues, power=2) #0.3449
np.sqrt(mean_tweedie_deviance(dsoyb_fit_elim['Y'], s_fit_elimg.fittedvalues, power=2)) #0.3828

#calculate AIC and BIC for Gamma
s_aic = s_fit_elimg.aic 
s_bic = s_fit_elimg.bic_llf
#LogLik: -504840; AIC: 1021116; BIC: 1021306


'''
Validate the model against the validation dataset
'''

#select the independent variables from the validation dataset
s_val_elim = dsoyb_val_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#let the model predict yield values for the validation dat
s_pred_elimg = s_fit_elimg.predict(s_val_elim)

#calculate McFadden's roh² and the RMGD scores
d2_tweedie_score(dsoyb_val_elim['Y'], s_pred_elimg, power=2) #0.3431
np.sqrt(mean_tweedie_deviance(dsoyb_val_elim['Y'], s_pred_elimg, power=2)) #0.3863


'''
Plot the Residuals for the model
'''
### Extract necessary measures ###

#select the independent variables from the fit dataset
s_fit_elim = dsoyb_fit_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#get the influence of the GLM model
s_stat_elimg = s_fit_elimg.get_influence()

#store cook's distance in a variable
s_elimg_cook = pd.Series(s_stat_elimg.cooks_distance[0]).transpose()
s_elimg_cook = s_elimg_cook.rename("Cooks_d", errors="raise")

#store the actual yield, the fitted values on response and link scale, 
#the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
#the absolute value of the resp and the sqrt of the stud residuals in a dataframe
#reset the index but keep the old one as a column in order to combine the dataframe
#with Cook's distance
s_data_infl = { 'Yield': dsoyb_fit_elim['Y'],
                'GLM_fitted': s_fit_elimg.fittedvalues,
               'Fitted_link': s_fit_elimg.predict(s_fit_elim, linear=True),
               'resid_pear': s_fit_elimg.resid_pearson, 
               'resid_stud' : s_stat_elimg.resid_studentized,
               'resid_resp_abs' : np.abs(s_fit_elimg.resid_response),
               'resid_stud_sqrt' : np.sqrt(np.abs(s_stat_elimg.resid_studentized)),
               'hat_matrix':s_stat_elimg.hat_matrix_diag}
s_elimg_infl = pd.DataFrame(data=s_data_infl).reset_index()
s_elimg_infl = pd.concat([s_elimg_infl, s_elimg_cook], axis='columns')

#take a sample of the influence dataframe to plot the lowess line
s_elimg_infl_sample = s_elimg_infl.sample(frac=0.1, random_state=2705)

### Studentized residuals vs. fitted values on link scale ###

#set plot characteristics
plot_ss = plt.figure(4)
plot_ss.set_figheight(8)
plot_ss.set_figwidth(12)

#Draw a scatterplot of studentized residuals vs. fitted values on the link scale
#lowess=True draws a fitted line which shows if the relationship is linear
plot_ss.axes[0] = sb.regplot('Fitted_link', 'resid_stud', data=s_elimg_infl_sample, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
#plt.scatter('Fitted_link', 'resid_stud', data=s_elimg_infl)

#plot labels
plot_ss.axes[0].set_title('Studentized Residuals vs Fitted on link scale')
plot_ss.axes[0].set_xlabel('Fitted values on link scale')
plot_ss.axes[0].set_ylabel('Studentized Residuals')

### Response residuals vs. fitted values on the response scale ###

#set plot characteristics
plot_sr = plt.figure(4)
plot_sr.set_figheight(8)
plot_sr.set_figwidth(12)

#Draw a scatterplot of response residuals vs. fitted values on the response scale
plot_sr.axes[0] = sb.residplot('GLM_fitted', 'Yield', data=s_elimg_infl, 
                          #lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#plot labels
plot_sr.axes[0].set_title('Residuals vs Fitted')
plot_sr.axes[0].set_xlabel('Fitted values')
plot_sr.axes[0].set_ylabel('Residuals')

#annotations of the three largest residuals
abs_resid = s_elimg_infl['resid_resp_abs'].sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_sr.axes[0].annotate(i, 
                               xy=(s_elimg_infl['GLM_fitted'][i], 
                                   s_elimg_infl['resid_resp_abs'][i]))

### QQ-Plot for the studentized residuals ###

#Specifications of the QQ Plot
QQ = ProbPlot(s_elimg_infl['resid_stud'], dist=stats.gamma, fit=True)
plot_sq = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

#set plot characteristics
plot_sq.set_figheight(8)
plot_sq.set_figwidth(12)

#plot labels
plot_sq.axes[0].set_title('Normal Q-Q')
plot_sq.axes[0].set_xlabel('Theoretical Quantiles')
plot_sq.axes[0].set_ylabel('Standardized Residuals');

#annotations of the three largest residuals
abs_norm_resid = np.flip(np.argsort(np.abs(s_elimg_infl['resid_stud'])), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_sq.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   s_elimg_infl['resid_stud'][i]));

### Cook's distance plots ###

##Cook's distance vs. no of observation##

#sort cook's distance value to get the value for the largest distance####
s_cook_sort = s_elimg_cook.sort_values(ascending=False)
#select all Cook's distance values which are greater than 4/n (n=number of datapoints)
s_cook_infl = s_elimg_cook.loc[s_elimg_cook > (4/(62237-21))].sort_values(ascending=False)

#barplot for values with the strongest influence (=largest Cook's distance)
#because running the function on all values takes a little longer
plt.bar(s_cook_infl.index, s_cook_infl)
plt.ylim(0, 0.01)

#plots for the ones greater than 4/n and all distance values
plt.scatter(s_cook_infl.index, s_cook_infl)
plt.scatter(s_elimg_cook.index, s_elimg_cook)
plt.ylim(0, 0.01)

##Studentized Residuals vs. Leverage w. Cook's distance line##

#set plot characteristics
plot_sc = plt.figure(4)
plot_sc.set_figheight(8)
plot_sc.set_figwidth(12)

#Draw the scatterplott of the Studentized residuals vs. leverage
plt.scatter(s_elimg_infl['hat_matrix'], s_elimg_infl['resid_stud'], alpha=0.5)
sb.regplot(s_elimg_infl['hat_matrix'], s_elimg_infl['resid_stud'], 
            scatter=False, 
            ci=False, 
            #lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#plot boundaries and labels
plot_sc.axes[0].set_xlim(0, 0.004)
plot_sc.axes[0].set_ylim(-3, 21)
plot_sc.axes[0].set_title('Residuals vs Leverage')
plot_sc.axes[0].set_xlabel('Leverage')
plot_sc.axes[0].set_ylabel('Standardized Residuals')

#annotate the three points with the largest Cooks distance value
leverage_top_3 = np.flip(np.argsort(s_elimg_infl["Cooks_d"]), 0)[:3]

for i in leverage_top_3:
    plot_sc.axes[0].annotate(i, 
                               xy=(s_elimg_infl['hat_matrix'][i], 
                                   s_elimg_infl['resid_stud'][i]))


###########################################################################
################ Loss of Industry Modelling ###############################
###########################################################################

'''
Prepare and modify datasets according to the assumptions of the LoI scenario
'''

#take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
#and available manure correctly
LoI_selim = ds0_raw

LoI_selim['mechanized'] = LoI_selim['mechanized'].replace(-9,np.nan)
LoI_selim['pesticides_H'] = LoI_selim['pesticides_H'].replace(-9,np.nan)

### Mechanised ###

#set mechanization to 0 in phase 2; due to the estimated stock in  fuel the variable remains 
#unchanged in phase 1
LoI_selim['mechanized_y2'] = LoI_selim['mechanized'].replace(1,0)

### N fertilizer ###

#drop all cells where mechanized or pesticiedes AND n_fertilizer are no data values
sn_drop= LoI_selim[((LoI_selim['mechanized'].isna())|(LoI_selim['pesticides_H'].isna()))
                & (LoI_selim['n_fertilizer']<0)].index
LoI_selim_pn = LoI_selim.drop(sn_drop)

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_selim_pn.loc[LoI_selim_pn['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan
LoI_selim_pn.loc[LoI_selim_pn['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
LoI_selim_pn[['n_fertilizer','p_fertilizer']] = LoI_selim_pn[['n_fertilizer','p_fertilizer']].fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
LoI_selim_pn.loc[LoI_selim_pn['n_total'] < 0, 'n_total'] = LoI_selim_pn['n_fertilizer'] + LoI_selim_pn['n_manure']

#drop the outliers (99.9th percentile) in the n and p fertilizer columns
LoI_selim_pn = LoI_selim_pn.loc[LoI_selim_pn['n_fertilizer'] < ds0_qt.iloc[6,4]]
LoI_selim_pn = LoI_selim_pn.loc[LoI_selim_pn['p_fertilizer'] < ds0_qt.iloc[6,5]]

#in phase 1, there will probably be a slight surplus of N (production>application)
#the surplus is assumed to be the new total

#calculate kg N applied per cell
LoI_selim_pn['n_kg'] = LoI_selim_pn['n_fertilizer']*LoI_selim_pn['area']
#calculate the fraction of the total N applied to soyb fields for each cell
LoI_selim_pn['n_ffrac'] = LoI_selim_pn['n_kg']/(LoI_selim_pn['n_kg'].sum())

#calculate the fraction of total N applied to soyb fields on the total N applied
#divide total of soyb N by 1000000 to get from kg to thousand t
s_nfert_frac = (LoI_selim_pn['n_kg'].sum())/1000000/118763
#calculate the new total for N soyb in phase one based on the N total surplus
s_ntot_new = s_nfert_frac * 14477 * 1000000

#calculate the new value of N application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_selim_pn['n_fert_y1'] = (s_ntot_new * LoI_selim_pn['n_ffrac']) / LoI_selim_pn['area']

#in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_selim_pn['n_fert_y2'] = 0

### P Fertilizer ###

#in phase 1, there will probably be a slight surplus of P (production>application)
#calculate kg p applied per cell
LoI_selim_pn['p_kg'] = LoI_selim_pn['p_fertilizer']*LoI_selim_pn['area']
#calculate the fraction of the total N applied to soyb fields for each cell
LoI_selim_pn['p_ffrac'] = LoI_selim_pn['p_kg']/(LoI_selim_pn['p_kg'].sum())

#calculate the fraction of total P applied to soyb fields on the total P applied to cropland
#divide total of soyb P by 1000000 to get from kg to thousand t
s_pfert_frac = (LoI_selim_pn['p_kg'].sum())/1000000/45858
#calculate the new total for P soyb in phase one based on the P total surplus
s_ptot_new = s_pfert_frac * 4142 * 1000000

#calculate the new value of P application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_selim_pn['p_fert_y1'] = (s_ptot_new * LoI_selim_pn['p_ffrac']) / LoI_selim_pn['area']

#in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_selim_pn['p_fert_y2'] = 0

############# N Manure ###################

#drop the rows containing outliers (99th percentile) in the manure column
LoI_selim_man = LoI_selim.loc[LoI_selim['n_manure'] < ds0_qt.iloc[5,6]]

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_selim_man['labor'] = LoI_selim_man['area']/5 #current value (7.4) is taken from Cole et al. (2016)
#due to information from a farmer, the value is set at 5

#calculate mean excretion rate of each cow in one year: cattle supplied ~ 43.7% of 131000 thousand t
#manure production in 2014, there were ~ 1.008.570.000(Statista)/1.439.413.930(FAOSTAT) 
#heads of cattle in 2014
cow_excr = 131000000000*0.437/1439413930

#calculate the new value of man application rate in kg per ha per cell, according
#to the available cows in each cell due to labor demand
LoI_selim_man['man_fert'] = (cow_excr * LoI_selim_man['labor']) / LoI_selim_man['area']

#that leads the application rate being the same in every cell because everybody has the same number of cows per ha
#it's assumed to be the same for both phases

### N total ###

#in phase 1, the total N available is the sum of available fertilizer and manure
LoI_selim['N_toty1'] = LoI_selim_pn['n_fert_y1'] + LoI_selim_man['man_fert']

#in phase 2 there is no more artificial fertilizer, so N total is equal to man_fert

### Pesticides ###

#drop the cells containing NaN values and outliers
LoI_selimp = LoI_selim.loc[LoI_selim['pesticides_H'].notna()]
LoI_selimp = LoI_selimp.loc[LoI_selim['pesticides_H'] < ds0_qt.iloc[6,9]]#~11

#in phase 1, there will probably be a slight surplus of Pesticides (production>application)

#calculate kg p applied per cell
LoI_selimp['pest_kg'] = LoI_selimp['pesticides_H']*LoI_selimp['area']
#calculate the fraction of the total N applied to soyb fields for each cell
LoI_selimp['pest_frac'] = LoI_selimp['pest_kg']/(LoI_selimp['pest_kg'].sum())

#calculate the fraction of total pesticides applied to soyb fields on the total pesticides applied to cropland
#divide total of soyb pesticides by 1000 to get from kg to t
s_pest_frac = (LoI_selimp['pest_kg'].sum())/1000/4190985

#due to missing reasonable data on the pesticide surplus, it is assumed that the
#surplus is in the same range as for P and N fertilizer
#the mean of N and P fertilizer surplus is calculated
frac_pest = ((14477/118763) + (4142/45858))/2
#calculate the new total for pesticides soyb in phase one based on the pesticides total surplus
s_pestot_new = s_pest_frac * (4190985 * frac_pest) * 1000

#calculate the new value of pesticides application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_selimp['pest_y1'] = (s_pestot_new * LoI_selimp['pest_frac']) / LoI_selimp['area']

#in phase 2 no industrially produced pesticides will be available anymore: set to 0
LoI_selimp['pest_y2'] = 0


### Irrigation ###

#in LoI it is assumed that only irrigation which is not reliant on electricity
#can still be maintained
#calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
#(fraction of irrigated cropland reliant on electricity)
LoI_selim['irr_LoI'] = LoI_selim['irrigation_tot'] * (1- LoI_selim['irrigation_rel'])

### Combine the different dataframes and drop rows with missing values ###

LoI_selim = pd.concat([LoI_selim, LoI_selim_pn['n_fert_y1'], LoI_selim_pn['n_fert_y2'],
                       LoI_selim_pn['p_fert_y1'], LoI_selim_pn['p_fert_y2'],
                       LoI_selim_man['man_fert'], LoI_selimp['pest_y1'], 
                       LoI_selimp['pest_y2']], axis='columns')

#Eliminate the rows without data:
LoI_selim = LoI_selim.dropna()

#Eliminating all points above the 99.9th percentile
LoI_selim = LoI_selim.loc[LoI_selim['Y'] < ds0_qt.iloc[6,3]]
LoI_selim = LoI_selim.loc[LoI_selim['n_total'] < ds0_qt.iloc[6,8]]


'''
Prediction of LoI yields and yield change rates in phase 1 and 2
'''
### Phase 1 ###

#select the rows from LoI_selim which contain the independent variables for phase 1
LoI_s_year1 = LoI_selim.iloc[:,[10,13,14,15,17,19,22,25]]
#reorder the columns according to the order in ds0_elim
LoI_s_year1 = LoI_s_year1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_s_year1 = LoI_s_year1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                      'pest_y1':"pesticides_H",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")

#predict the yield for phase 1 using the gamma GLM
s_yield_y1 = s_fit_elimg.predict(LoI_s_year1)
#calculate the change rate from actual yield to the predicted yield
s_y1_change = ((s_yield_y1-soyb_kgha)/soyb_kgha).dropna()
#calculate the number of cells with a postivie change rate
s1 = s_y1_change.loc[s_y1_change>0]

#create a new variable with the yields for positive change rates set to orginial yields
s01 = s_y1_change.loc[s_y1_change > 0]
s_y1_0 = LoI_selim['Y']
s_y1_0 = s_y1_0[s01.index]
s011 = s_y1_change.loc[s_y1_change <= 0]
s_y1_1 = s_yield_y1[s011.index]
s_y1_y0 = s_y1_0.append(s_y1_1)

#calculate statistics for yield and change rate

#calculate weights for mean change rate calculation dependent on current yield
#and current soybean area in a cell
sw=LoI_selim['Y']*ds0_elim['area']
sw = sw.fillna(method='ffill')

#calculate weighted mean, min and max of predicted yield (1) including postive change rates
smean_y1_weigh = round(np.average(s_yield_y1, weights=LoI_selim['area']),2) #2257.13kg/ha
smax_y1 = s_yield_y1.max() #3508.91 kg/ha
smin_y1 = s_yield_y1.min() #675.42 kg/ha
#(2) excluding postive change rates
smean_y1_0 = round(np.average(s_y1_y0, weights=LoI_selim['area']),2) #1923.81kg/ha
smax_y10 = s_y1_y0.max()  # 3250.71.1kg/ha
smin_y10 = s_y1_y0.min()  # 54.89kg/ha

#change rate
smean_y1c_weigh = round(np.average(s_y1_change, weights=sw),2) #-0.04 (~-4%) -0.1 (~10%)
smax_y1c = s_y1_change.max() # +27.78 (~+2780%)
smin_y1c = s_y1_change.min() #-0.8256 (~-83%)

### Phase 2 ###

#select the rows from LoI_selim which contain the independent variables for phase 2
LoI_s_year2 = LoI_selim.iloc[:,[13,14,15,16,19,23,24,26]]
#reorder the columns according to the order in ds0_elim
LoI_s_year2 = LoI_s_year2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 
                       'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
#rename the columns according to the names used in the model formular
LoI_s_year2 = LoI_s_year2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                      'pest_y2':"pesticides_H",'mechanized_y2':"mechanized",
                                      'irr_LoI':"irrigation_tot"}, errors="raise")

#predict the yield for phase 2 using the gamma GLM
s_yield_y2 = s_fit_elimg.predict(LoI_s_year2)
#calculate the change from actual yield to the predicted yield
s_y2_change = ((s_yield_y2-soyb_kgha)/soyb_kgha).dropna()
#calculate the number of cells with a postivie change rate
s2 = s_y2_change.loc[s_y2_change>0]

#create a new variable with all positive change rates set to 0 for both phases
s_c0 = pd.concat([s_y1_change, s_y2_change], axis=1)
s_c0 = s_c0.rename(columns={0:"s_y1_c0", 1:"s_y2_c0"}, errors="raise")
s_c0.loc[s_c0['s_y1_c0'] > 0, 's_y1_c0'] = 0
s_c0.loc[s_c0['s_y2_c0'] > 0, 's_y2_c0'] = 0

#create a new variable with the yields for positive change rates set to orginial yields
s02 = s_y2_change.loc[s_y2_change > 0]
s_y2_0 = LoI_selim['Y']
s_y2_0 = s_y2_0[s02.index]
s022 = s_y2_change.loc[s_y2_change <= 0]
s_y2_1 = s_yield_y2[s022.index]
s_y2_y0 = s_y2_0.append(s_y2_1)

#calculate statistics for yield and change rate

#calculate weighted mean, min and max of predicted yield (1) including postive change rates
smean_y2_weigh = round(np.average(s_yield_y2, weights=LoI_selim['area']),2) #1600.52 kg/ha
smax_y2 = s_yield_y2.max() #2442.53kg/ha
smin_y2 = s_yield_y2.min() #675.35kg/ha
#(2) excluding postive change rates
smean_y2_0 = round(np.average(s_y2_y0, weights=LoI_selim['area']),2) #1536.79 kg/ha
smax_y20 = s_y2_y0.max()  # 2442.53.1kg/ha
smin_y20 = s_y2_y0.min()  # 54.89kg/ha

#calculate weighted mean, min and max of predicted change rate (1) including postive change rates
smean_y2c_weigh = round(np.average(s_y2_change, weights=sw),2) #-0.3 (~-30%) -0.36 (~36%)
smax_y2c = s_y2_change.max() #27.77 (~+2780%)
smin_y2c = s_y2_change.min() #-0.8271 (~-83%)
#(2) excluding postive change rates
smean_y2c0_weigh = round(np.average(s_c0['s_y2_c0'], weights=sw),2) #-0.38
smean_y1c0_weigh = round(np.average(s_c0['s_y1_c0'], weights=sw),2) #-0.17

'''
Statistics to compare current SPAM2010 yield with (1) current fitted values,
(2) phase 1 and (3) phase 2 predictions
'''

## calculate statistics for current yield ##

#SPAM2010 yield: weighted mean, max, min, total yearly production
ds0_mean = round(np.average(ds0_elim['Y'], weights=ds0_elim['area']),2)
ds0_max = ds0_elim['Y'].max()
ds0_min = ds0_elim['Y'].min()
ds0_prod = (ds0_elim['Y'] * ds0_elim['area']).sum()
#fitted values for current yield based on Gamma GLM: wieghted mean, max and min
s_fit_mean = round(np.average(s_fit_elimg.fittedvalues, weights=ds0_elim['area']),2)
s_fit_max =s_fit_elimg.fittedvalues.max()
s_fit_min = s_fit_elimg.fittedvalues.min()
s_fit_prod = (s_fit_elimg.fittedvalues * ds0_elim['area']).sum()

## calculate statistics for both phases ##

#phase 1: calculate the percentage of current yield/production will be achieved
#in phase 1 as predicted by the GLM, calculate total production in phase 1
#(1) including positive change rates and 
s_y1_per = smean_y1_weigh/ds0_mean #~71.7% of current average yield
s_y1_prod = (s_yield_y1 * LoI_selim['area']).sum()
#(2) with positive change rates set to 0
s_y10_prod = (s_y1_y0 * LoI_selim['area']).sum()
s_y10_per = s_y10_prod/ds0_prod

#phase 2: calculate the percentage of current yield/production will be achieved
#in phase 2 as predicted by the GLM, calculate total production in phase 1
#(1) including positive change rates and 
s_y2_per = smean_y2_weigh/ds0_mean #67.3% of current average yield
s_y2_prod = (s_yield_y2 * LoI_selim['area']).sum()
#(2) with positive change rates set to 0
s_y20_prod = (s_y2_y0 * LoI_selim['area']).sum()
s_y20_per = s_y20_prod/ds0_prod

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
print(ds0_mean, s_fit_mean, smean_y1_weigh, smean_y2_weigh,
      ds0_prod, s_fit_prod, s_y1_prod, s_y2_prod,
      ds0_max, s_fit_max, smax_y1, smax_y2, s_fit_max,
      ds0_min,  s_fit_min, smin_y1, smin_y2,
      s_y1_per, s_y2_per, s_y10_per, s_y20_per,
      smean_y1c_weigh, smean_y2c_weigh, smean_y1c0_weigh, smean_y2c0_weigh)


'''
save the predicted yields and the yield change rates for each phase
'''

#combine yields and change rates of both phases with the latitude and longitude values
LoI_soyb = pd.concat([soyb_yield['lats'], soyb_yield['lons'], s_yield_y1,
                       s_y1_change, s_yield_y2, s_y2_change, s_c0], axis='columns')
LoI_soyb = LoI_soyb.rename(columns={0:"s_yield_y1", 1:"s_y1_change", 
                                      2:"s_yield_y2",3:"s_y2_change"}, errors="raise")
#save the dataframe in a csv
LoI_soyb.to_csv(params.geopandasDataDir + "LoIsoybYieldHighRes.csv")

#save the yield for phase 1 and 2 and the change rate for phase 1 and 2 with and without positive rates
#as ASCII files
utilities.create5minASCIIneg(LoI_soyb,'s_y1_change',params.asciiDir+'LoISoybYieldChange_y1')
utilities.create5minASCIIneg(LoI_soyb,'s_yield_y1',params.asciiDir+'LoISoybYield_y1')
utilities.create5minASCIIneg(LoI_soyb,'s_y2_change',params.asciiDir+'LoISoybYieldChange_y2')
utilities.create5minASCIIneg(LoI_soyb,'s_yield_y2',params.asciiDir+'LoISoybYield_y2')
utilities.create5minASCIIneg(LoI_soyb,'s_y1_c0',params.asciiDir+'LoISoybYieldChange_0y1')
utilities.create5minASCIIneg(LoI_soyb,'s_y2_c0',params.asciiDir+'LoISoybYieldChange_0y2')
