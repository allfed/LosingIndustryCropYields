'''

File containing the code to explore data and perform a multiple regression
on yield for crop
'''

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src import params
from src import utilities
# from src import outdoor_growth
# from src.outdoor_growth import OutdoorGrowth
from src import stat_ut
import pandas as pd
import scipy
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#seaborn is just used for plotting, might be removed later
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.metrics import r2_score
from sklearn.metrics import d2_tweedie_score
from sys import platform

if platform == "linux" or platform == "linux2":
    #this is to ensure Morgan's computer doesn't crash
    import resource
    rsrc = resource.RLIMIT_AS
    resource.setrlimit(rsrc, (4e9, 4e9))#no more than 3 gb
    # soft_limit,hard_limit=resource.getrlimit(resource.RLIMIT_DATA)
    # print('soft_limit')
    # print(soft_limit)
    # print('hard_limit')
    # print(hard_limit)
    # for name, desc in [
    #     ('RLIMIT_CORE', 'core file size'),
    #     ('RLIMIT_CPU',  'CPU time'),
    #     ('RLIMIT_FSIZE', 'file size'),
    #     ('RLIMIT_DATA', 'heap size'),
    #     ('RLIMIT_STACK', 'stack size'),
    #     ('RLIMIT_RSS', 'resident set size'),
    #     ('RLIMIT_NPROC', 'number of processes'),
    #     ('RLIMIT_NOFILE', 'number of open files'),
    #     ('RLIMIT_MEMLOCK', 'lockable memory address'),
    #     ]:
    #     limit_num = getattr(resource, name)
    #     soft, hard = resource.getrlimit(limit_num)
    #     print('Maximum %-25s (%-15s) : %20s %20s' % (desc, name, soft, hard))
    # quit()
import gc


params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for crop
IRRIGATED_AND_RAINFED = True
IRRIGATED=True
CROP = 'wheat'
SAVE_DOWNSAMPLES = True
crop_dict = {'maize':{'pesticides':'Corn','yield':'MAIZ'},\
    'wheat':{'pesticides':'Wheat','yield':'WHEA'},\
    'soybean':{'pesticides':'Soybean','yield':'SOYB'},\
    'rice':{'pesticides':'Rice','yield':'RICE'}\
    }

if(IRRIGATED_AND_RAINFED):
    postfix = ''
elif(IRRIGATED):
    postfix = 'Irr'
else:
    postfix = "Rf"

crop_yield=pd.read_csv(params.geopandasDataDir + crop_dict[CROP]['yield'] + 'CropYield'+postfix+'Filtered.csv')

#select all rows from crop_yield for which the column growArea has a value greater than zero
crop_nozero=crop_yield.loc[crop_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
crop_kgha=crop_nozero['yield_kgPerHa']

crop_kgha_log=np.log(crop_kgha)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot crop yield distribution in a histogram
# plt.hist(crop_kgha, bins=50)
# plt.title('Maize yield ha/kg')
# plt.xlabel('yield kg/ha')
# plt.ylabel('density')


#plot log transformed values of yield_kgPerHa
# plt.hist(crop_kgha_log, bins=50)
# plt.show()

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

pesticides=pd.read_csv(params.geopandasDataDir +crop_dict[CROP]['pesticides']+ 'Pesticides'+postfix+'Filtered.csv')
fertilizer=pd.read_csv(params.geopandasDataDir + 'Fertilizer'+postfix+'Filtered.csv') #kg/m²
fertilizer_man=pd.read_csv(params.geopandasDataDir + 'FertilizerManure'+postfix+'Filtered.csv') #kg/km²
crop = pd.read_csv(params.geopandasDataDir + 'FracCropArea'+postfix+'Filtered.csv')
if(IRRIGATED):
    irr_rel=pd.read_csv(params.geopandasDataDir + 'FracReliant'+postfix+'Filtered.csv')
    irr_t=pd.read_csv(params.geopandasDataDir + 'FracIrrigationArea'+postfix+'Filtered.csv')
tillage=pd.read_csv(params.geopandasDataDir + 'TillageAllCrops'+postfix+'Filtered.csv')
aez=pd.read_csv(params.geopandasDataDir + 'AEZ'+postfix+'Filtered.csv')

#fraction of irrigation total is of total cell area so I have to divide it by the
#fraction of crop area in a cell and set all values >1 to 1
if(IRRIGATED):
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
data_raw = {"lat": crop_yield.loc[:,'lats'],
        "lon": crop_yield.loc[:,'lons'],
        "area": crop_yield.loc[:,'growArea'],
        "Y": crop_yield.loc[:,'yield_kgPerHa'],
        "n_fertilizer": fertilizer.loc[:,'n_kgha'],
        "p_fertilizer": fertilizer.loc[:,'p_kgha'],
        "n_manure": fertilizer_man.loc[:,'applied_kgha'],
        "n_man_prod" : fertilizer_man.loc[:,'produced_kgha'],
        "n_total" : N_total,
        "pesticides_H": pesticides.loc[:,'total_H'],
        "mechanized": tillage.loc[:,'is_mech'],
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
        }
if(IRRIGATED):
    data_raw['irrigation_tot'] = irr_tot
    data_raw['irrigation_rel'] = irr_rel.loc[:,'frac_reliant']

#arrange data_raw in a dataframe
dcrop_raw = pd.DataFrame(data=data_raw)
#select only the rows where the area of the cropland is larger than 0
d0_raw=dcrop_raw.loc[dcrop_raw['area'] > 0]

d0_raw['pesticides_H'] = d0_raw['pesticides_H'].replace(np.nan, -9)
if(IRRIGATED):
    d0_raw['irrigation_rel'] = d0_raw['irrigation_rel'].replace(np.nan, 0)

#replace 0s in the moisture, climate and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
d0_raw['thz_class'] = d0_raw['thz_class'].replace(0,np.nan)
d0_raw['mst_class'] = d0_raw['mst_class'].replace(0,np.nan)
d0_raw['soil_class'] = d0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
d0_raw['thz_class'] = d0_raw['thz_class'].replace([8,9,10],7)
d0_raw['mst_class'] = d0_raw['mst_class'].replace(2,1)
d0_raw['mst_class'] = d0_raw['mst_class'].replace(7,6)

#fill in the NaN vlaues in the dataset with a forward filling method
#(replacing NaN with the value in the cell before)
d0_raw = d0_raw.fillna(method='ffill')

#Handle the data by eliminating the rows without data:
d0_elim = d0_raw.loc[d0_raw['pesticides_H'] > -9]
d0_elim = d0_elim.loc[d0_raw['mechanized'] > -9] 
#replace remaining no data values in the fertilizer datasets with NaN and then fill them
d0_elim.loc[d0_elim['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
d0_elim.loc[d0_elim['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
d0_elim = d0_elim.fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
d0_elim.loc[d0_elim['n_total'] < 0, 'n_total'] = d0_elim['n_fertilizer'] + d0_elim['n_manure']

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
d0_qt = d0_elim.quantile([.1, .25, .5, .75, .8,.85, .87, .9, .95,.975, .99,.995, .999,.9999])
d0_qt.reset_index(inplace=True, drop=True)
d0_elim = d0_elim.loc[d0_elim['Y'] < d0_qt.iloc[12,3]] #~12500
d0_elim = d0_elim.loc[d0_elim['n_fertilizer'] < d0_qt.iloc[12,4]]#~180
d0_elim = d0_elim.loc[d0_elim['p_fertilizer'] < d0_qt.iloc[12,5]] #~34
d0_elim = d0_elim.loc[d0_elim['n_manure'] < d0_qt.iloc[10,6]] #~11
#d0_elim = d0_elim.loc[d0_elim['n_man_prod'] < d0_qt.iloc[12,7]] #~44
d0_elim = d0_elim.loc[d0_elim['n_total'] < d0_qt.iloc[12,8]] #~195
d0_elim = d0_elim.loc[d0_elim['pesticides_H'] < d0_qt.iloc[12,9]]#~11

#drop all rows with an area below 100 ha
#d0_elim=d0_elim.loc[d0_elim['area'] > 100]


'''
plt.scatter(d0_elim["p_fertilizer"], d0_elim["Y"])

plt.hist(d0_elim['n_manure'], bins=50)


#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('d0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=d0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=d0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=d0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=d0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=d0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=d0_elim, x='Y')

ax = sb.boxplot(x=d0_elim["irrigation_tot"])
ax = sb.boxplot(x=d0_elim["n_man_prod"])
ax = sb.boxplot(x="mechanized", y='Y', data=d0_elim)
ax = sb.boxplot(x="thz_class", y='Y', hue='mechanized', data=d0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=d0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=d0_elim)
'''
#mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########
dum_mst = pd.get_dummies(d0_elim['mst_class'])
dum_thz = pd.get_dummies(d0_elim['thz_class'])
dum_soil = pd.get_dummies(d0_elim['soil_class'])
#####Rename Columns##########
dum_mst = dum_mst.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
dum_thz = dum_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
dum_soil = dum_soil.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
dcrop_d_elim = pd.concat([d0_elim, dum_mst, dum_thz, dum_soil], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
if(IRRIGATED):
    dcrop_dum_elim = dcrop_d_elim.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')
else:
    dcrop_dum_elim = dcrop_d_elim.drop(['270+days','Temp_cool+Bor+Arctic'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dcrop_val_elim = dcrop_dum_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dcrop_fit_elim = dcrop_dum_elim.drop(dcrop_val_elim.index) #RAW
##################Collinearity################################

#caution takes up a lot of memory
#sb.pairplot(dcrop_fit_elim)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
if(IRRIGATED):
    dcrop_cor_elim = dcrop_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
else:
    dcrop_cor_elim = dcrop_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                        'thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
#calculate spearman (rank transformed) correlation coeficcients between the 
#independent variables and save the values in a dataframe
sp = dcrop_cor_elim.corr(method='spearman')
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

X = add_constant(dcrop_cor_elim)
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

'''
const                38.602255
p_fertilizer          5.199395
n_total               6.325616
pesticides_H          2.413725
mechanized            1.627253
irrigation_tot        1.926472
LGP<120days           1.474690
120-180days           1.698274
180-225days           1.635152
225-270days           1.443363
Trop_low              3.113578
Trop_high             1.423234
Sub-trop_warm         1.611023
Sub-trop_mod_cool     1.656092
Sub-trop_cool         1.435898
Temp_mod              1.576973
S1_very_steep         1.384381
S2_hydro_soil         1.322476
S3_no-slight_lim      3.896556
S4_moderate_lim       3.959483
S5_severe_lim         1.779372
dtype: float64
'''

######################Regression##############################

#R-style formula

#determine models
#Normal distribution
#m_mod_elimn = smf.ols(formula=' Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized +  C(thz_class) + \
#              C(mst_class) + C(soil_class) ', data=dcrop_fit_elim)
#Gamma distribution
if(IRRIGATED):
    # mod_elimg = smf.glm(formula='Y ~ n_total + mechanized + \
    #           C(thz_class) + C(mst_class) + C(soil_class)', data=dcrop_fit_elim, 
    #           family=sm.families.Gamma(link=sm.families.links.log))
    mod_elimg = smf.glm(formula='Y ~ n_total + mechanized + irrigation_tot + p_fertilizer + pesticides_H + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dcrop_fit_elim, 
               family=sm.families.Gamma(link=sm.families.links.log))

else:
    mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dcrop_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))

#Nullmodel
mod_elim0 = smf.glm(formula='Y ~ 1', data=dcrop_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
#m_fit_elimn = mod_elimn.fit()
fit_elimg = mod_elimg.fit()
fit_elim0 = mod_elim0.fit()
#print results
#print(fit_elimn.summary())
#LogLik: -2547000; AIC: 5094000; BIC: 5094000
print('fit_elimg.summary()')
print(fit_elimg.summary())
print('fit_elim0.summary()')
print(fit_elim0.summary())


###########Fit statistics#############
#calculate pseudo R² for the Gamma distribution
# pseudoR_elim = 1-(121800/196880) #0.38135
#m_pseudoR_elim = 1-(53432/102030) # 0.4763 without the cells above 100 ha
# print('pseudoR_elim')
# print(pseudoR_elim)

#calculate AIC and BIC for Gamma
aic = fit_elimg.aic 
bic = fit_elimg.bic_llf
#LogLik: -2479200; AIC: 4958456; BIC: 4958677 (-3305574: this doesn't track)

########Validation against the validation dataset########

#select the independent variables from the val dataset
if(IRRIGATED):
    independent_columns=['p_fertilizer','n_total','pesticides_H','irrigation_rel','irrigation_tot','mechanized','thz_class','mst_class','soil_class']
else:
    independent_columns=['p_fertilizer','n_total','pesticides_H','mechanized','thz_class','mst_class','soil_class']

independent_column_indices = []
for i in np.arange(0,len(dcrop_val_elim.columns)):
    val_elim = dcrop_val_elim.iloc[:,[i]]
    column = dcrop_val_elim.columns[i]
    if(column in independent_columns):
        independent_column_indices.append(i)


val_elim = dcrop_val_elim.iloc[:,independent_column_indices]

#fit the model against the validation data
#pred_elim = fit_elimn.predict(val_elim)
pred_elimg = fit_elimg.predict(val_elim)


# print(dcrop_val_elim[])
# print(pred_elimg)
# quit()

# ################### Change Resolution ##################
print("r2_score(dcrop_val_elim['Y'], pred_elimg)") #0.3572
print(r2_score(dcrop_val_elim['Y'], pred_elimg)) #0.3572
print("roh2_score(dcrop_val_elim['Y'], pred_elimg)")
print(d2_tweedie_score(dcrop_val_elim['Y'], pred_elimg, power=2))

five_minute = 5/60

#return an array at lower resolution yield weighted by area
def downsample(highres_data,highres_prediction,scale):
    step = five_minute*scale # degrees 
    to_bin = lambda x: np.floor(x / step) * step
    highres_data["latbin"] = highres_data['lat'].map(to_bin)
    highres_data["lonbin"] = highres_data['lon'].map(to_bin)
    groups = highres_data.groupby(["latbin", "lonbin"])

    # print('by area')
    highres_data['predicted'] = highres_prediction
    highres_data['sumproduct']=highres_data['Y']*highres_data['area']
    highres_data['sumproduct_pred']=highres_data['predicted']*highres_data['area']

    lowres = pd.DataFrame({})
    lowres['area']=highres_data.groupby(["latbin", "lonbin"]).area.sum()
    lowres['sumproduct']=highres_data.groupby(["latbin", "lonbin"]).sumproduct.sum()
    lowres['sumproduct_pred']=highres_data.groupby(["latbin", "lonbin"]).sumproduct_pred.sum()
    lowres['mean']= lowres['sumproduct'] / lowres['area']
    lowres['mean_pred']= lowres['sumproduct_pred'] / lowres['area']

    # print("len(lowres['mean'])")
    # print("len(lowres['mean_pred'])")

    # print(len(lowres['mean']))
    # print(len(lowres['mean_pred']))

    del highres_data['sumproduct_pred']
    del highres_data['sumproduct']
    del highres_data['predicted']

    return lowres

r2_scores = []
five_minute = 5/60 #degrees
for scale in np.arange(1,250,5):
    lowres = downsample(dcrop_val_elim,pred_elimg,scale)
    r2_scores.append(d2_tweedie_score(lowres['mean'], lowres['mean_pred'], power=2))
                     #(r2_score(lowres['mean'],lowres['mean_pred']))

plt.figure()
x = np.linspace(5/60,len(r2_scores)*5/60,len(r2_scores))
plt.scatter(x,r2_scores)

#pesticides crop names are capital letter first, so I chose that option
plt.title(crop_dict[CROP]['pesticides'] + ' Variable Resolution Validation roh^2')


# x = np.linspace(1,len(r2_scores),int(len(r2_scores)/5))
# plt.xticks(x, np.linspace(5/60,len(r2_scores)*5/60,int(len(r2_scores)/5)))
plt.xlabel('resolution, degrees')
plt.ylabel('roh^2 between predicted and validation data')
# plt.show()
# print("r2_score(lowres['mean'], lowres['mean_pred'])") #0.3572
# print(r2_score(lowres['mean'],lowres['mean_pred'])) #0.3572
#.49432 without cells below 100ha

'''
#plot the predicted against the observed values
plt.scatter(pred_elim, dcrop_val_elim['Y'])
plt.scatter(pred_elimg, dcrop_val_elim['Y'])
plt.scatter(dcrop_val_elim['n_total'], pred_elimg)
plt.scatter(dcrop_val_elim['n_total'], dcrop_val_elim['Y'])

#plot the histogram
plt.hist(pred_elimg, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')


plt.hist(dcrop_val_elim['Y'], bins=50)

an_elim = pd.concat([dcrop_val_elim, pred_elim, pred_elimg], axis='columns')
an_elim = an_elim.rename(columns={0:"pred_elim", 1:"pred_elimg"}, errors="raise")
sb.lmplot(x='pred_elimg', y='Y', data=an_elim)
'''



##########RESIDUALS for the Gamma distribution#############


#select the independent variables from the fit dataset
fit_elim = dcrop_fit_elim.iloc[:,independent_column_indices]

#get the influence of the GLM model
stat_elimg = fit_elimg.get_influence()
#print(stat_elimg.summary_table()), there seems to be too much data

#store cook's distance in a variable
elimg_cook = pd.Series(stat_elimg.cooks_distance[0]).transpose()
elimg_cook = elimg_cook.rename("Cooks_d", errors="raise")

#store the actual yield, the fitted values on response and link scale, 
#the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
#the absolute value of the resp and the sqrt of the stud residuals in a dataframe
#reset the index but keep the old one as a column in order to combine the dataframe
#with Cook's distance
data_infl = { 'Yield': dcrop_fit_elim['Y'],
                'GLM_fitted': fit_elimg.fittedvalues,
               'Fitted_link': fit_elimg.predict(fit_elim, linear=True),
               'resid_pear': fit_elimg.resid_pearson, 
               'resid_stud' : stat_elimg.resid_studentized,
               'resid_resp_abs' : np.abs(fit_elimg.resid_response),
               'resid_stud_sqrt' : np.sqrt(np.abs(stat_elimg.resid_studentized)),
               'hat_matrix':stat_elimg.hat_matrix_diag}
elimg_infl = pd.DataFrame(data=data_infl).reset_index()
elimg_infl = pd.concat([elimg_infl, elimg_cook], axis='columns')


#take a sample of the influence dataframe to plot the lowess line
elimg_infl_sample = elimg_infl.sample(frac=0.2, random_state=3163)



##########Residual Plot############

#########Studentized residuals vs. fitted values on link scale######

# plot_elimg = plt.figure(4)
# plot_elimg.set_figheight(8)
# plot_elimg.set_figwidth(12)
# plt.scatter('Fitted_link', 'resid_stud', data=elimg_infl)
# plot_elimg.axes[0].set_title('Studentized Residuals vs Fitted on link scale')
# plot_elimg.axes[0].set_xlabel('Fitted values on link scale')
# plot_elimg.axes[0].set_ylabel('Studentized Residuals')
# plt.show()

#########Response residuals vs. fitted values on response scale#######
# plot_elimg = plt.figure(4)
# plot_elimg.set_figheight(8)
# plot_elimg.set_figwidth(12)


# plot_elimg.axes[0] = sb.residplot('GLM_fitted', 'Yield', data=elimg_infl, 
#                           #lowess=True, 
#                           scatter_kws={'alpha': 0.5}, 
#                           line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

# plot_elimg.axes[0].set_title('Residuals vs Fitted')
# plot_elimg.axes[0].set_xlabel('Fitted values')
# plot_elimg.axes[0].set_ylabel('Residuals')

# # annotations
# abs_resid = elimg_infl['resid_resp_abs'].sort_values(ascending=False)
# abs_resid_top_3 = abs_resid[:3]

# for i in abs_resid_top_3.index:
#     plot_elimg.axes[0].annotate(i, 
#                                xy=(elimg_infl['GLM_fitted'][i], 
#                                    elimg_infl['resid_resp_abs'][i]))

# plt.show()
###############QQ-Plot########################

# plt.figure()
# QQ = ProbPlot(elimg_infl['resid_stud'])
# plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

# plot_lm_2.set_figheight(8)
# plot_lm_2.set_figwidth(12)

# plot_lm_2.axes[0].set_title('Normal Q-Q')
# plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
# plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# # annotations
# abs_norm_resid = np.flip(np.argsort(np.abs(elimg_infl['resid_stud'])), 0)
# abs_norm_resid_top_3 = abs_norm_resid[:3]

# for r, i in enumerate(abs_norm_resid_top_3):
#     plot_lm_2.axes[0].annotate(i, 
#                                xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
#                                    elimg_infl['resid_stud'][i]));
# plt.show()

############Cook's distance plot##########

#############Cook's distance vs. no of observation######
# plt.figure()
# #sort cook's distance value to get the value for the largest distance####
# cook_sort = elimg_cook.sort_values(ascending=False)
# #select all Cook's distance values which are greater than 4/n (n=number of datapoints)
# cook_infl = elimg_cook.loc[elimg_cook > (4/273772)].sort_values(ascending=False)

# #barplot for values with the strongest influence (=largest Cook's distance)
# #because running the function on all values takes a little longer
# plt.bar(cook_infl.index, cook_infl)
# plt.ylim(0, 0.01)

# #plots for largest 3 cook values, the ones greater than 4/n and all distance values
# plt.scatter(cook_infl.index[0:3], cook_infl[0:3])
# plt.scatter(cook_infl.index, cook_infl)
# plt.scatter(elimg_cook.index, elimg_cook)
# plt.ylim(0, 0.01)
# plt.show()
############Studentized Residuals vs. Leverage w. Cook's distance line#####

# plot_lm_4 = plt.figure(4)
# plot_lm_4.set_figheight(8)
# plot_lm_4.set_figwidth(12)

# plt.scatter(elimg_infl['hat_matrix'], elimg_infl['resid_stud'], alpha=0.5)
# sb.regplot(elimg_infl['hat_matrix'], elimg_infl['resid_stud'], 
#             scatter=False, 
#             ci=False, 
#             #lowess=True,
#             line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# plot_lm_4.axes[0].set_xlim(0, 0.004)
# plot_lm_4.axes[0].set_ylim(-3, 21)
# plot_lm_4.axes[0].set_title('Residuals vs Leverage')
# plot_lm_4.axes[0].set_xlabel('Leverage')
# plot_lm_4.axes[0].set_ylabel('Standardized Residuals')
# plt.show()

# annotate the three points with the largest Cooks distance value
# leverage_top_3 = np.flip(np.argsort(elimg_infl["Cooks_d"]), 0)[:3]

# for i in leverage_top_3:
#     plot_elimg.axes[0].annotate(i, 
#                                xy=(elimg_infl['hat_matrix'][i], 
#                                    elimg_infl['resid_stud'][i]))

# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(fit_elimg.params) # number of model parameters

# graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
#       np.linspace(0.001, 0.200, 50), 
#       'Cook\'s distance') # 0.5 line
# graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
#       np.linspace(0.001, 0.200, 50)) # 1 line
# plt.legend(loc='upper right');
# plt.show()

    

'''
#index of leverage top 3 is not the index of the datapoints, therefore I print
#the elimg_infl rows at this index because it contains the old index as a column
for i in leverage_top_3:
    print(elimg_infl.iloc[i])

sm.graphics.plot_regress_exog(fit_elimg, 'n_total')
plt.show()


'''


#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

#take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
#and available manure correctly
LoI_celim = d0_raw

LoI_celim['mechanized'] = LoI_celim['mechanized'].replace(-9,np.nan)
LoI_celim['pesticides_H'] = LoI_celim['pesticides_H'].replace(-9,np.nan)

############ Mechanised ##########################

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_celim['mechanized_y2'] = LoI_celim['mechanized'].replace(1,0)

############ N fertilizer #########################

mn_drop= LoI_celim[((LoI_celim['mechanized'].isna())|(LoI_celim['pesticides_H'].isna()))
                & (LoI_celim['n_fertilizer']<0)].index
LoI_celim_pn = LoI_celim.drop(mn_drop)

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_celim_pn.loc[LoI_celim_pn['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
LoI_celim_pn.loc[LoI_celim_pn['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
LoI_celim_pn[['n_fertilizer','p_fertilizer']] = LoI_celim_pn[['n_fertilizer','p_fertilizer']].fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
LoI_celim_pn.loc[LoI_celim_pn['n_total'] < 0, 'n_total'] = LoI_celim_pn['n_fertilizer'] + LoI_celim_pn['n_manure']

#drop the nonsense values (99.9th percentile) in the n and p fertilizer columns
LoI_celim_pn = LoI_celim_pn.loc[LoI_celim_pn['n_fertilizer'] < d0_qt.iloc[12,4]]#~180
LoI_celim_pn = LoI_celim_pn.loc[LoI_celim_pn['p_fertilizer'] < d0_qt.iloc[12,5]] #~34

#in year 1, there will probably be a slight surplus of N (production>application)
#calculate kg N applied per cell
LoI_celim_pn['n_kg'] = LoI_celim_pn['n_fertilizer']*LoI_celim_pn['area']
#calculate the fraction of the total N applied to crop fields for each cell
LoI_celim_pn['n_ffrac'] = LoI_celim_pn['n_kg']/(LoI_celim_pn['n_kg'].sum())

#calculate the fraction of total N applied to crop fields on the total N applied
#divide total of crop N by 1000000 to get from kg to thousand t
nfert_frac = (LoI_celim_pn['n_kg'].sum())/1000000/118763
#calculate the new total for N crop in year one based on the N total surplus
ntot_new = nfert_frac * 14477 * 1000000

#calculate the new value of N application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_celim_pn['n_fert_y1'] = (ntot_new * LoI_celim_pn['n_ffrac']) / LoI_celim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_celim_pn['n_fert_y2'] = 0
#LoI_celim_pn.loc[LoI_celim_pn['n_fert_y2'] > 0, 'n_fert_y1'] = 0

############## P Fertilizer #####################

#in year 1, there will probably be a slight surplus of P (production>application)
#calculate kg p applied per cell
LoI_celim_pn['p_kg'] = LoI_celim_pn['p_fertilizer']*LoI_celim_pn['area']
#calculate the fraction of the total N applied to crop fields for each cell
LoI_celim_pn['p_ffrac'] = LoI_celim_pn['p_kg']/(LoI_celim_pn['p_kg'].sum())

#calculate the fraction of total P applied to crop fields on the total P applied to cropland
#divide total of crop P by 1000000 to get from kg to thousand t
pfert_frac = (LoI_celim_pn['p_kg'].sum())/1000000/45858
#calculate the new total for P crop in year one based on the P total surplus
ptot_new = pfert_frac * 4142 * 1000000

#calculate the new value of P application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_celim_pn['p_fert_y1'] = (ptot_new * LoI_celim_pn['p_ffrac']) / LoI_celim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_celim_pn['p_fert_y2'] = 0

############# N Manure ###################

#drop the rows containing nonsense values (99th percentile) in the manure column
LoI_celim_man = LoI_celim.loc[LoI_celim['n_manure'] < d0_qt.iloc[10,6]] #~11

#calculate kg N applied per cell: 1,018,425,976.75 kg total
LoI_celim_man['man_kg'] = LoI_celim_man['n_manure']*LoI_celim_man['area']
#calculate the fraction of the total N applied to crop fields for each cell
LoI_celim_man['n_mfrac'] = LoI_celim_man['man_kg']/(LoI_celim_man['man_kg'].sum())

#calculate the fraction of total N applied to crop fields of the total N applied to cropland
#divide total of crop N by 1000000 to get from kg to thousand t
nman_frac = (LoI_celim_man['man_kg'].sum())/1000000/24000

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_celim_man['labor'] = LoI_celim_man['area']/5 #current value (7.4) is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer
#I chose 5 now just because I don't believe 7.4 is correct

#calculate mean excretion rate of each cow in one year: cattle supplied ~ 43.7% of 131000 thousand t
#manure production in 2014, there were ~ 1.008.570.000(Statista)/1.439.413.930(FAOSTAT) 
#heads of cattle in 2014
cow_excr = 131000000000*0.437/1439413930

#calculate available manure based on cow labor demand: 1,278,868,812.065 kg
man_av = cow_excr * LoI_celim_man['labor'].sum()
#more manure avialable then currently applied, but that is good as N from mineral
#fertilizer will be missing

#calculate the new value of man application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_celim_man['man_fert'] = (man_av * LoI_celim_man['n_mfrac']) / LoI_celim_man['area']


########### N total ######################

LoI_celim['N_toty1'] = LoI_celim_pn['n_fert_y1'] + LoI_celim_man['man_fert']
#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_celim['area_LoI'] = LoI_celim['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations

############## Pesticides #####################

LoI_celimp = LoI_celim.loc[LoI_celim['pesticides_H'].notna()]
LoI_celimp = LoI_celimp.loc[LoI_celimp['pesticides_H'] < d0_qt.iloc[12,9]]#~11
#in year 1, there will probably be a slight surplus of Pesticides (production>application)
#calculate kg p applied per cell
LoI_celimp['pest_kg'] = LoI_celimp['pesticides_H']*LoI_celimp['area']
#calculate the fraction of the total N applied to crop fields for each cell
LoI_celimp['pest_frac'] = LoI_celimp['pest_kg']/(LoI_celimp['pest_kg'].sum())

#calculate the fraction of total pesticides applied to crop fields on the total pesticides applied to cropland
#divide total of crop pesticides by 1000 to get from kg to t
pest_frac = (LoI_celimp['pest_kg'].sum())/1000/4190985

#due to missing reasonable data on the pesticide surplus, it is assumed that the
#surplus is in the same range as for P and N fertilizer
frac_pest = ((14477/118763) + (4142/45858))/2
#calculate the new total for pesticides crop in year one based on the pesticides total surplus
pestot_new = pest_frac * (4190985 * frac_pest) * 1000

#calculate the new value of pesticides application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_celimp['pest_y1'] = (pestot_new * LoI_celimp['pest_frac']) / LoI_celimp['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_celimp['pest_y2'] = 0


########## Irrigation ####################

#in LoI it is assumed that only irrigation which is not reliant on electricity
#can still be maintained
#calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
#(fraction of irrigated cropland reliant on electricity)
if(IRRIGATED):
    LoI_celim['irr_LoI'] = LoI_celim['irrigation_tot'] * (1- LoI_celim['irrigation_rel'])

###########Combine the different dataframes and drop rows with missing values#########

LoI_celim = pd.concat([LoI_celim, LoI_celim_pn['n_fert_y1'], LoI_celim_pn['n_fert_y2'],
                       LoI_celim_pn['p_fert_y1'], LoI_celim_pn['p_fert_y2'],
                       LoI_celim_man['man_fert'], LoI_celimp['pest_y1'], 
                       LoI_celimp['pest_y2']], axis='columns')

#Handle the data by eliminating the rows without data:
LoI_celim = LoI_celim.dropna()

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
#d0_qt = d0_elim.quantile([.1, .25, .5, .75, .8,.85, .87, .9, .95,.975, .99,.995, .999,.9999])
#d0_qt.reset_index(inplace=True, drop=True)
LoI_celim = LoI_celim.loc[LoI_celim['Y'] < d0_qt.iloc[12,3]] #~12500
#d0_elim = d0_elim.loc[d0_elim['n_man_prod'] < d0_qt.iloc[12,7]] #~44
LoI_celim = LoI_celim.loc[LoI_celim['n_total'] < d0_qt.iloc[12,8]] #~195

# quit()

#########################Prediction of LoI yields#########################

################## Year 1 ##################

#select the rows from LoI_celim which contain the independent variables for year 1

# independent_columns=['thz_class','mst_class','soil_class','120-180days','225-270days','Sub-trop_warm','Temp_mod']
independent_columns = ['mechanized', 'thz_class', 'mst_class', 'soil_class',
    'N_toty1','irr_LoI', 'p_fert_y1', 'pest_y1']

independent_column_indices = []
for i in np.arange(0,len(LoI_celim.columns)):
    val_elim = LoI_celim.iloc[:,[i]]
    column = LoI_celim.columns[i]
    if(column in independent_columns):
        independent_column_indices.append(i)


LoI_year1 = LoI_celim.iloc[:,independent_column_indices]

#reorder the columns according to the order in d0_elim
if(IRRIGATED):
    LoI_year1 = LoI_year1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                           'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_year1 = LoI_year1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                          'pest_y1':"pesticides_H",
                                          'irr_LoI':"irrigation_tot"}, errors="raise")
else:
    LoI_year1 = LoI_year1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                       'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_year1 = LoI_year1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                          'pest_y1':"pesticides_H"}, errors="raise")
#predict the yield for year 1 using the gamma GLM
yield_y1 = fit_elimg.predict(LoI_year1)
#calculate the change rate from actual yield to the predicted yield
y1_change = (yield_y1-crop_kgha)/crop_kgha

#calculate statistics for yield and change rate

#yield
mmean_y1_weigh = round(np.average(yield_y1, weights=LoI_celim['area']),2) #3832.02kg/ha
mmax_y1 = yield_y1.max() #10002.44 kg/ha
mmin_y1 = yield_y1.min() #691.74 kg/ha

#change rate
#mmean_y1c_weigh = round(np.average(y1_change, weights=crop_yield['growArea']),2)
mmax_y1c = y1_change.max() # +105.997 (~+10600%)
mmin_y1c = y1_change.min() #-0.94897 (~-95%)

################## Year 2 ##################

#select the rows from LoI_celim which contain the independent variables for year 2
# independent_columns=['thz_class','mst_class','soil_class','LGP<120days','225-270days','Sub-trop_mod_cool','Sub-trop_cool','S1_very_steep']
independent_columns=['thz_class', 'mst_class', 'soil_class', 'mechanized_y2', 'irr_LoI',
       'p_fert_y2', 'man_fert', 'pest_y2']

independent_column_indices = []
for i in np.arange(0,len(LoI_celim.columns)):
    val_elim = LoI_celim.iloc[:,[i]]
    column = LoI_celim.columns[i]
    if(column in independent_columns):
        independent_column_indices.append(i)

LoI_year2 = LoI_celim.iloc[:,independent_column_indices]#reorder the columns according to the order in d0_elim
if(IRRIGATED):
    LoI_year2 = LoI_year2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 
                           'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_year2 = LoI_year2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                          'pest_y2':"pesticides_H",'mechanized_y2':"mechanized",
                                          'irr_LoI':"irrigation_tot"}, errors="raise")
else:
    LoI_year2 = LoI_year2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_year2 = LoI_year2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                          'pest_y2':"pesticides_H",'mechanized_y2':"mechanized"}, errors="raise")    

#predict the yield for year 2 using the gamma GLM
yield_y2 = fit_elimg.predict(LoI_year2)
#calculate the change from actual yield to the predicted yield
y2_change = (yield_y2-crop_kgha)/crop_kgha

c_c0 = pd.concat([y1_change, y2_change], axis=1)
c_c0 = c_c0.rename(columns={0: "y1_change0", 1: "y2_change0"}, errors="raise")
c_c0.loc[c_c0['y1_change0'] > 0, 'y1_change0'] = 0
c_c0.loc[c_c0['y2_change0'] > 0, 'y2_change0'] = 0

#calculate statistics for yield and change rate

#yield
mmean_y2_weigh = round(np.average(yield_y2, weights=LoI_celim['area']),2) #2792.08kg/ha
mmax_y2 = yield_y2.max() #6551.74kg/ha
mmin_y2 = yield_y2.min() #689.79kg/ha

#change rate
mmean_y2c = y2_change.mean() #0.1198 (~+12%)
mmax_y2c = y2_change.max() #70.087 (~+7000%)
mmin_y2c = y2_change.min() #-0.9503 (~-95%)

#combine both yields and change rates with the latitude and longitude values
# LoI = pd.concat([crop_yield['lats'], crop_yield['lons'], yield_y1,
                # y1_change, yield_y2, y2_change], axis='columns')
data = {\
    'index':crop_yield['index'],\
    'lats':crop_yield['lats'],\
    'lons':crop_yield['lons'],\
    'area':LoI_celim['area'],\
    'yield_y1': yield_y1,\
    'yield_y2':yield_y2,\
    'y1_change':y1_change,\
    'y2_change':y2_change,\
    'y1_change0':c_c0['y1_change0'],\
    'y2_change0':c_c0['y2_change0']\
}
LoI = pd.DataFrame(data=data)

#return an array at lower resolution yield weighted by area
def downsample_prediction_1col(highres_data,data_col, step):
    #drop any nan values from yields
    # highres_data.dropna(subset = [data_col], inplace=True)
    #set any nan area to 0 weight
    highres_data[['area']] = highres_data[['area']].fillna(value=0)

    #any nan valued data_col rows will be set to zero area 
    highres_data.loc[np.isnan(highres_data[data_col]), 'area'] = 0

    #set the now zero area rows with nan values to be negative, so they can be filtered later
    # this will have zero weighting
    highres_data[[data_col]] = highres_data[[data_col]].fillna(value=-9) 
    gc.collect()

    to_bin = lambda x: np.floor(x / step) * step
    print("highres_data['lats'] final")
    # print(highres_data['lats'])
    highres_data["latbin"] = highres_data['lats'].map(to_bin)
    # del highres_data['lats']
    highres_data["lonbin"] = highres_data['lons'].map(to_bin)
    # del highres_data['lons']
    groups = highres_data.groupby(["latbin", "lonbin"])

    highres_data[data_col]=highres_data[data_col]*highres_data['area']

    lowres = pd.DataFrame({})
    lowres['area']=highres_data.groupby(["latbin", "lonbin"]).area.sum()
    lowres['sumproduct_'+data_col]=highres_data.groupby(["latbin", "lonbin"])[data_col].sum()

    del highres_data[data_col]

    lowres['area'].replace(0, np.NaN)

    # lowres['index'] = lowres.index.values
    lowres[data_col]= lowres['sumproduct_'+data_col] / lowres['area']
    del lowres['sumproduct_'+data_col]

    #set zero area rows to nan data_col 
    lowres.loc[lowres['area']==0] = np.nan

    lowres = lowres.reset_index()
    lowres.rename(columns={'lonbin':'lons', 'latbin':'lats'}, inplace=True)
    lowres['lons'] = lowres['lons']+180
    lowres['lons'][lowres['lons']>180] = lowres['lons']-360
    lowres.sort_values(by=['lats', 'lons'],inplace=True)

    return lowres

def create_high_res_latlon():
    five_minute = 5/60
    # we ignore the last latitude cell
    lats = np.linspace(-90, 90 - five_minute, \
                        np.floor(180 / five_minute).astype('int'))
    lons = np.linspace(-180, 180 - five_minute, \
                        np.floor(360 / five_minute).astype('int'))

    # time2 = datetime.datetime.now()

    lats2d, lons2d = np.meshgrid(lats, lons)
    data = {"lats": pd.Series(lats2d.ravel()),
            "lons": pd.Series(lons2d.ravel())}
    # time3 = datetime.datetime.now()
    df = pd.DataFrame(data=data)

    return df

if(SAVE_DOWNSAMPLES):

    #2 degree seems reasonable
    scale = 24
    step = five_minute*scale # 5/60 = 0.0833, 24 * 0.0833 degrees = 2 degrees

    # raw_data=pd.read_csv(params.geopandasDataDir + crop_dict[CROP]['gap']+'YieldGap'+postfix+'HighRes.csv')

    high_res=create_high_res_latlon()
    # print(high_res['lats'])
    #clear out some ram to leave room for future arrays 
    for i in dir():
        if isinstance(globals()[i], pd.DataFrame):
            if(i == "LoI_celim" or i == "LoI_celim_man" or i == "LoI_celim_pn" or i == "LoI_celimp" or i == "LoI_year1" or i == "LoI_year2" or i == "X" or i == "aez" or i == "crop" or i == "crop_nozero" or i == "crop_yield" or i == "d0_elim" or i == "d0_raw" or i == "dcrop_cor_elim" or i == "dcrop_d_elim" or i == "dcrop_dum_elim" or i == "dcrop_fit_elim" or i == "dcrop_raw" or i == "dcrop_val_elim" or i == "dum_mst" or i == "dum_soil" or i == "dum_thz" or i == "elimg_infl" or i == "elimg_infl_sample" or i == "fert_new" or i == "fertilizer" or i == "fertilizer_man" or i == "fit_elim" or i == "irr_rel" or i == "irr_t" or i == "man_new" or i == "pesticides" or i == "tillage" or i == "val_eli"):
                del globals()[i]
    gc.collect()


    high_res['area'] = np.nan
    high_res.iloc[LoI['index'],high_res.columns.get_loc('area')] = LoI['area'].values
    print('saving downsample columns')

    downsample_columns = [
        'yield_y1',\
        'yield_y2',\
        'y1_change',\
        'y2_change',\
        'y1_change0',\
        'y2_change0']
    for dc in downsample_columns:
        high_res[dc] = np.nan
        high_res.iloc[LoI['index'],high_res.columns.get_loc(dc)] = LoI[dc].values

        # nan_array = create_nan_array(step)
        # lowres = downsample_prediction(raw_data,step)
        # print(high_res['lats'])
        lowres = downsample_prediction_1col(high_res,dc,step)

        # del high_res[dc]

        #skip CSV generation
        # lowres.to_csv(params.geopandasDataDir + "LoI"+crop_dict[CROP]['yield']+"Yield"+postfix+"LowRes.csv")

        utilities.createASCII(lowres,dc,params.asciiDir+crop_dict[CROP]['yield']+'LoI_'+dc)
        print('saved '+dc)
else:
    #save the dataframe in a csv
    print('saving high res')
    raw_data=pd.read_csv(params.geopandasDataDir + crop_dict[CROP]['yield']+'CropYield'+postfix+'HighRes.csv')
    
    downsample_columns = [
        'yield_y1',\
        'yield_y2',\
        'y1_change',\
        'y2_change',\
        'y1_change0',\
        'y2_change0']
    for dc in downsample_columns:
        raw_data[dc] = np.nan
        raw_data.iloc[LoI['index'],raw_data.columns.get_loc(dc)] = LoI[dc].values
        raw_data.to_csv(params.geopandasDataDir + crop_dict[CROP]['yield']+'LOIYield'+dc+'HighRes.csv')
        del raw_data[dc]
        print('saved '+dc)

    print('saved')