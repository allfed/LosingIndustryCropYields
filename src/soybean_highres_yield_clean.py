'''

File containing the code to explore data and perform a multiple regression
on yield for soyb
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

#import yield geopandas data for soyb

soyb_yield=pd.read_csv(params.geopandasDataDir + 'SOYBCropYieldHighRes.csv')

#select all rows from soyb_yield for which the column growArea has a value greater than zero
soyb_nozero=soyb_yield.loc[soyb_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
soyb_kgha=soyb_nozero['yield_kgPerHa']

soyb_kgha_log=np.log(soyb_kgha)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


#plot soyb yield distribution in a histogram
plt.hist(soyb_kgha, bins=50)
plt.title('soyb yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

#plot log transformed values of yield_kgPerHa
plt.hist(soyb_kgha_log, bins=50)


'''
Fitting of distributions to the data and comparing the fit
'''

'''
#calculate loglik, AIC & BIC for each distribution
st = stat_ut.stat_overview(dist_lists, pdf_lists, param_dicts)

    Distribution  loglikelihood           AIC           BIC
7  normal on log  -3.748763e+05  7.497686e+05  7.498546e+05
6  Inverse Gamma  -2.859205e+06  5.718427e+06  5.718513e+06
5          Gamma  -2.860227e+06  5.720469e+06  5.720555e+06
3         normal  -2.908569e+06  5.817154e+06  5.817240e+06
1    exponential  -2.910045e+06  5.820106e+06  5.820192e+06
0        lognorm  -3.587555e+06  7.175125e+06  7.175211e+06
2        weibull  -3.694327e+06  7.388671e+06  7.388757e+06
4     halfnormal           -inf           inf           inf
Best fit is normal on log by far, then inverse gamma on non-log
'''

'''
Load factor data and extract zeros
'''
s_pesticides=pd.read_csv(params.geopandasDataDir + 'SoybeanPesticidesHighRes.csv')
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
#select only the rows where the area of the cropland is larger than 0
ds0_raw=dsoyb_raw.loc[dsoyb_raw['area'] > 0]

ds0_raw['pesticides_H'] = ds0_raw['pesticides_H'].replace(np.nan, -9)
ds0_raw['irrigation_rel'] = ds0_raw['irrigation_rel'].replace(np.nan, -9)

#test mech dataset values
s_test_mech0 = ds0_raw.loc[ds0_raw['mechanized'] == 0] #82541, now 95508
s_test_mech1 = ds0_raw.loc[ds0_raw['mechanized'] == 1] #172097, now 196854
s_test_mechn = ds0_raw.loc[ds0_raw['mechanized'] == -9] #90658, now 52934
#this is a problem: -9 is used as NaN value and there are way, way too many

s_test_f = ds0_raw.loc[ds0_raw['n_fertilizer'] < 0] #11074 0s, 4205 NaNs
s_test_pf = ds0_raw.loc[ds0_raw['p_fertilizer'] < 0] #11770 0s, 4205 NaNs
s_test_man = ds0_raw.loc[ds0_raw['n_manure'] < 0] #9794 0s, 0 NaNs
s_test_p = ds0_raw.loc[ds0_raw['pesticides_H'] < 0] #183822 NaNs but no 0s

#replace 0s in the moisture, climate and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
ds0_raw['thz_class'] = ds0_raw['thz_class'].replace(0,np.nan)
ds0_raw['mst_class'] = ds0_raw['mst_class'].replace(0,np.nan)
ds0_raw['soil_class'] = ds0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
ds0_raw['thz_class'] = ds0_raw['thz_class'].replace([8,9,10],7)
ds0_raw['mst_class'] = ds0_raw['mst_class'].replace(2,1)
ds0_raw['mst_class'] = ds0_raw['mst_class'].replace(7,6)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
ds0_raw = ds0_raw.fillna(method='ffill')

#Handle the data by eliminating the rows without data:
ds0_elim = ds0_raw.loc[ds0_raw['pesticides_H'] > -9]
ds0_elim = ds0_elim.loc[ds0_raw['mechanized'] > -9] 

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
ds0_elim.loc[ds0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
ds0_elim.loc[ds0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
ds0_elim = ds0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
ds0_elim.loc[ds0_elim['n_total'] < 0, 'n_total'] = ds0_elim['n_fertilizer'] + ds0_elim['n_manure']

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
ds0_qt = ds0_elim.quantile([.1, .25, .5, .75, .8, .95, .999,.9999])
ds0_qt.reset_index(inplace=True, drop=True)
ds0_elim = ds0_elim.loc[ds0_elim['Y'] < ds0_qt.iloc[6,3]]
ds0_elim = ds0_elim.loc[ds0_elim['n_fertilizer'] < ds0_qt.iloc[6,4]]
ds0_elim = ds0_elim.loc[ds0_elim['p_fertilizer'] < ds0_qt.iloc[6,5]]
ds0_elim = ds0_elim.loc[ds0_elim['n_manure'] < ds0_qt.iloc[6,6]]
ds0_elim = ds0_elim.loc[ds0_elim['n_man_prod'] < ds0_qt.iloc[6,7]]
ds0_elim = ds0_elim.loc[ds0_elim['n_total'] < ds0_qt.iloc[6,8]]
ds0_elim = ds0_elim.loc[ds0_elim['pesticides_H'] < ds0_qt.iloc[6,9]]


#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('ds0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=ds0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=ds0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=ds0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=ds0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=ds0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=ds0_elim, x='Y')

ax = sb.boxplot(x=ds0_elim["irrigation_tot"])
ax = sb.boxplot(x=ds0_elim["irrigation_rel"])
ax = sb.boxplot(x="mechanized", y='Y', data=ds0_elim)
ax = sb.boxplot(x="thz_class", y='Y', hue='mechanized', data=ds0_elim)
ax = sb.boxplot(x="mst_class", y='Y', data=ds0_elim)
ax = sb.boxplot(x="soil_class", y='Y', data=ds0_elim)

#mst, thz and soil are categorical variables which need to be converted into 
#dummy variables before running the regression
dus_mst_elim = pd.get_dummies(ds0_elim['mst_class'])
dus_thz_elim = pd.get_dummies(ds0_elim['thz_class'])
dus_soil_elim = pd.get_dummies(ds0_elim['soil_class'])
#rename the columns according to the classes
dus_mst_elim = dus_mst_elim.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
dus_thz_elim = dus_thz_elim.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
dus_soil_elim = dus_soil_elim.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
dsoyb_d_elim = pd.concat([ds0_elim, dus_mst_elim, dus_thz_elim, dus_soil_elim], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
dsoyb_dus_elim = dsoyb_d_elim.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dsoyb_val_elim = dsoyb_dus_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dsoyb_fit_elim = dsoyb_dus_elim.drop(dsoyb_val_elim.index) #RAW

##################Collinearity################################


#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
dsoyb_cor_elim = dsoyb_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
#one method to calculate correlations but without the labels of the pertaining variables
#spearm = stats.spearmanr(dsoyb_cor_raw)
#calculates spearman (rank transformed) correlation coeficcients between the 
#independent variables and saves the values in a dataframe
sp_s = dsoyb_cor_elim.corr(method='spearman')
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

Xs = add_constant(dsoyb_cor_elim)
pd.Series([variance_inflation_factor(Xs.values, i) 
               for i in range(Xs.shape[1])], 
              index=Xs.columns)
'''
const                40.379846
p_fertilizer          7.244317
n_total               7.922601
pesticides_H          2.610407
mechanized            2.163687
irrigation_tot        1.955759
LGP<120days           1.146552
120-180days           1.667719
180-225days           1.558867
225-270days           1.500510
Trop_low              3.085479
Trop_high             1.246571
Sub-trop_warm         1.786493
Sub-trop_mod_cool     2.225860
Sub-trop_cool         1.586660
Temp_mod              1.935220
S1_very_steep         1.379096
S2_hydro_soil         1.305151
S3_no-slight_lim      3.785709
S4_moderate_lim       3.442216
S5_severe_lim         1.671891
dtype: float64
'''

######################Regression##############################

#R-style formula

#determine models
#Normal distribution
s_mod_elimn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
              C(mst_class) + C(soil_class) ', data=dsoyb_fit_elim)
#Gamma distribution
s_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dsoyb_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))
#Nullmodel
s_mod_elim0 = smf.glm(formula='Y ~ 1', data=dsoyb_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
s_fit_elimn = s_mod_elimn.fit()
s_fit_elimg = s_mod_elimg.fit()
s_fit_elim0 = s_mod_elim0.fit()
#print results
print(s_fit_elimn.summary()) #0.298
#LogLik: -970480; AIC: 1941000; BIC: 1941000
print(s_fit_elimg.summary())
print(s_fit_elim0.summary())


###########Fit statistics#############
#calculate pseudo R² for the Gamma distribution
s_pseudoR_elim = 1-(30255/41291) #0.2673
print(s_pseudoR_elim)

#calculate AIC and BIC for Gamma
s_aic = s_fit_elimg.aic 
s_bic = s_fit_elimg.bic_llf
#LogLik: -969090; AIC: 1938220; BIC: 1938423 (-3305574: this doesn't track)

########Validation against the validation dataset########

#select the independent variables from the val dataset
s_val_elim = dsoyb_val_elim.iloc[:,[5,8,9,10,11,13,14,15]]

#fit the model against the validation data
s_pred_elim = s_fit_elimn.predict(s_val_elim)
s_pred_elimg = s_fit_elimg.predict(s_val_elim)

#calculate the R² scores
r2_score(dsoyb_val_elim['Y'], s_pred_elim) #0.28605
r2_score(dsoyb_val_elim['Y'], s_pred_elimg) #0.27711

#plot the predicted against the observed values
plt.scatter(s_pred_elim, dsoyb_val_elim['Y'])
plt.scatter(s_pred_elimg, dsoyb_val_elim['Y'])

#plot the histogram
plt.hist(s_pred_elimg, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')

'''
plt.hist(dsoyb_val_elim['Y'], bins=50)

an_elim = pd.concat([dsoyb_val_elim, pred_elim, pred_elimg], axis='columns')
an_elim = an_elim.rename(columns={0:"pred_elim", 1:"pred_elimg"}, errors="raise")
sb.lmplot(x='pred_elimg', y='Y', data=an_elim)
'''



##########RESIDUALS#############

#######Gamma

#get the influence of the GLM model
s_stat_elimg = s_fit_elimg.get_influence()
#print(m_stat_elimg.summary_table()), there seems to be too much data
s_elimg_cook = pd.DataFrame(s_stat_elimg.cooks_distance).transpose()
s_elimg_cook = s_elimg_cook.rename(columns={0:"Cooks_d", 1:"ones"}, errors="raise")
s_data_infl = {'GLM_fitted': s_fit_elimg.fittedvalues,
       'hat_matrix':s_stat_elimg.hat_matrix_diag, 
       'resid_stud' : s_stat_elimg.resid_studentized}
s_elimg_infl = pd.DataFrame(data=s_data_infl).reset_index()
s_elimg_infl = pd.concat([s_elimg_infl, s_elimg_cook], axis='columns')
s_yiel_res =dsoyb_fit_elim['Y'].reset_index(drop=True)

s_mod_resid = s_fit_elimg.resid_response.reset_index(drop=True)
model_abs_resid = np.abs(s_mod_resid)
s_stud_sqrt = np.sqrt(np.abs(s_elimg_infl['resid_stud']))
s_resid = pd.concat([s_mod_resid, model_abs_resid, s_stud_sqrt], axis='columns')
s_resid = s_resid.rename(columns={0:"resid_pear", 1:"resid_pear_abs", 'resid_stud':"resid_stud_sqrt"}, errors="raise")

s_elimg_infl_sample = pd.concat([s_elimg_infl, s_resid, s_yiel_res], axis='columns')
s_elimg_infl_sample = s_elimg_infl_sample.sample(frac=0.2, random_state=2705)

##########Residual Plot############
plot_elimg = plt.figure(4)
plot_elimg.set_figheight(8)
plot_elimg.set_figwidth(12)


plot_elimg.axes[0] = sb.residplot(s_elimg_infl['GLM_fitted'], dsoyb_fit_elim['Y'], 
                          #lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_elimg.axes[0].set_title('Residuals vs Fitted')
plot_elimg.axes[0].set_xlabel('Fitted values')
plot_elimg.axes[0].set_ylabel('Residuals')

# annotations
s_abs_resid = s_mod_resid.sort_values(ascending=False)
s_abs_resid_top_3 = s_abs_resid[:3]

for i in s_abs_resid_top_3.index:
    plot_elimg.axes[0].annotate(i, 
                               xy=(s_elimg_infl['GLM_fitted'][i], 
                                   s_mod_resid[i]))

###############QQ-Plot########################

QQ = ProbPlot(s_elimg_infl['resid_stud'])
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# annotations
s_abs_norm_resid = np.flip(np.argsort(np.abs(s_elimg_infl['resid_stud'])), 0)
s_abs_norm_resid_top_3 = s_abs_norm_resid[:3]

for r, i in enumerate(s_abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   s_elimg_infl['resid_stud'][i]));

############Cook's distance plot##########

plot_lm_4 = plt.figure(4)
plot_lm_4.set_figheight(8)
plot_lm_4.set_figwidth(12)

plt.scatter(s_elimg_infl['hat_matrix'], s_elimg_infl['resid_stud'], alpha=0.5)
sb.regplot(s_elimg_infl['hat_matrix'], s_elimg_infl['resid_stud'], 
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
s_leverage_top_3 = np.flip(np.argsort(s_elimg_infl["Cooks_d"]), 0)[:3]

for i in s_leverage_top_3:
    plot_elimg.axes[0].annotate(i, 
                               xy=(s_elimg_infl['hat_matrix'][i], 
                                   s_elimg_infl['resid_stud'][i]))

# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(s_fit_elimg.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance') # 0.5 line
graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50)) # 1 line
plt.legend(loc='upper right');




'''
sm.graphics.plot_regress_exog(s_fit_elimg, 'n_total')
plt.show()


'''

#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

man_tot = fertilizer_man['applied_kgha'].loc[fertilizer_man['applied_kgha'] > -1].sum()
man_ptot = fertilizer_man['produced_kgha'].loc[fertilizer_man['produced_kgha'] > -1].sum()
per_man = man_tot/man_ptot *100

LoI_selim = ds0_elim.drop(['Y'], axis='columns')

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_selim['mechanized_y2'] = LoI_selim['mechanized'].replace(1,0)

#in year 1, there will probably be a slight surplus of N (production>application)
#divivde each cell of n_fertilizer with the sum of all cells and multiply with new total
LoI_selim['n_fert_y1'] = LoI_selim['n_fertilzer']/LoI_selim['n_fertilizer'].sum() * 14477
#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_selim.loc[LoI_selim['n_fert_y2'] > 0, 'n_fert_y1'] = 0
#LoI_mraw['n_fert_y1'] = LoI_mraw['n_fertilizer'].replace(1,0)

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_selim['labor'] = LoI_selim['area']/7.4 #current value is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer

#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_selim['area_LoI'] = LoI_selim['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations


LoI_selim['mechanized'] = LoI_selim['mechanized'].replace(1,0)
LoI_selim['labor'] = LoI_selim['area']/7.4


