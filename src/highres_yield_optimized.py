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
#seaborn is just used for plotting, might be removed later
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.metrics import r2_score
from sys import platform

if platform == "linux" or platform == "linux2":
    #this is to ensure Morgan's computer doesn't crash
    import resource
    rsrc = resource.RLIMIT_AS
    resource.setrlimit(rsrc, (3e9, 3e9))#no more than 3 gb



params.importAll()


'''
Import data, extract zeros and explore data statistic values and plots 
'''

#import yield geopandas data for maize
IRRIGATED_AND_RAINFED = True
IRRIGATED=True
CROP = 'MAIZ'

if(IRRIGATED):
    if(IRRIGATED_AND_RAINFED):
        postfix = ''
    else:
        postfix = 'Irr'
else:
    postfix = "Rf"

maize_yield=pd.read_csv(params.geopandasDataDir + CROP+'CropYield'+postfix+'Filtered.csv')

#select all rows from maize_yield for which the column growArea has a value greater than zero
maize_nozero=maize_yield.loc[maize_yield['growArea'] > 0]
#compile yield data where area is greater 0 in a new array
maize_kgha=maize_nozero['yield_kgPerHa']

maize_kgha_log=np.log(maize_kgha)

#sets design aspects for the following plots
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

#plot maize yield distribution in a histogram
# plt.hist(maize_kgha, bins=50)
# plt.title('Maize yield ha/kg')
# plt.xlabel('yield kg/ha')
# plt.ylabel('density')


#plot log transformed values of yield_kgPerHa
# plt.hist(maize_kgha_log, bins=50)
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

m_pesticides=pd.read_csv(params.geopandasDataDir + 'CornPesticides'+postfix+'Filtered.csv')
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
        "thz_class" : aez.loc[:,'thz'],
        "mst_class" : aez.loc[:,'mst'],
        "soil_class": aez.loc[:,'soil']
        }
if(IRRIGATED):
    datam_raw['irrigation_tot'] = irr_tot
    datam_raw['irrigation_rel'] = irr_rel.loc[:,'frac_reliant']

#arrange data_raw in a dataframe
dmaize_raw = pd.DataFrame(data=datam_raw)
#select only the rows where the area of the cropland is larger than 0
dm0_raw=dmaize_raw.loc[dmaize_raw['area'] > 0]

dm0_raw['pesticides_H'] = dm0_raw['pesticides_H'].replace(np.nan, -9)
if(IRRIGATED):
    dm0_raw['irrigation_rel'] = dm0_raw['irrigation_rel'].replace(np.nan, 0)

#replace 0s in the moisture, climate and soil classes as well as 7 & 8 in the
#soil class with NaN values so they can be handled with the .fillna method
dm0_raw['thz_class'] = dm0_raw['thz_class'].replace(0,np.nan)
dm0_raw['mst_class'] = dm0_raw['mst_class'].replace(0,np.nan)
dm0_raw['soil_class'] = dm0_raw['soil_class'].replace([0,7,8],np.nan)
#replace 9 & 10 with 8 to combine all three classes into one Bor+Arctic class
dm0_raw['thz_class'] = dm0_raw['thz_class'].replace([8,9,10],7)
dm0_raw['mst_class'] = dm0_raw['mst_class'].replace(2,1)
dm0_raw['mst_class'] = dm0_raw['mst_class'].replace(7,6)

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

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
dm0_qt = dm0_elim.quantile([.1, .25, .5, .75, .8,.85, .87, .9, .95,.975, .99,.995, .999,.9999])
dm0_qt.reset_index(inplace=True, drop=True)
dm0_elim = dm0_elim.loc[dm0_elim['Y'] < dm0_qt.iloc[12,3]] #~12500
dm0_elim = dm0_elim.loc[dm0_elim['n_fertilizer'] < dm0_qt.iloc[12,4]]#~180
dm0_elim = dm0_elim.loc[dm0_elim['p_fertilizer'] < dm0_qt.iloc[12,5]] #~34
dm0_elim = dm0_elim.loc[dm0_elim['n_manure'] < dm0_qt.iloc[10,6]] #~11
#dm0_elim = dm0_elim.loc[dm0_elim['n_man_prod'] < dm0_qt.iloc[12,7]] #~44
dm0_elim = dm0_elim.loc[dm0_elim['n_total'] < dm0_qt.iloc[12,8]] #~195
dm0_elim = dm0_elim.loc[dm0_elim['pesticides_H'] < dm0_qt.iloc[12,9]]#~11

#drop all rows with an area below 100 ha
#dm0_elim=dm0_elim.loc[dm0_elim['area'] > 100]


'''
plt.scatter(dm0_elim["p_fertilizer"], dm0_elim["Y"])

plt.hist(dm0_elim['n_manure'], bins=50)


#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('dm0_elim Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=dm0_elim, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=dm0_elim, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=dm0_elim, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=dm0_elim, x='n_total')
sb.boxplot(ax=axes[1, 1], data=dm0_elim, x='pesticides_H')
sb.boxplot(ax=axes[1, 2], data=dm0_elim, x='Y')

ax = sb.boxplot(x=dm0_elim["irrigation_tot"])
ax = sb.boxplot(x=dm0_elim["n_man_prod"])
ax = sb.boxplot(x="mechanized", y='Y', data=dm0_elim)
ax = sb.boxplot(x="thz_class", y='Y', hue='mechanized', data=dm0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Y', data=dm0_elim)
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Y', data=dm0_elim)
'''
#mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########
mdum_mst = pd.get_dummies(dm0_elim['mst_class'])
mdum_thz = pd.get_dummies(dm0_elim['thz_class'])
mdum_soil = pd.get_dummies(dm0_elim['soil_class'])
#####Rename Columns##########
mdum_mst = mdum_mst.rename(columns={1:"LGP<120days", 3:"120-180days", 4:"180-225days",
                                  5:"225-270days", 6:"270+days"}, errors="raise")
mdum_thz = mdum_thz.rename(columns={1:"Trop_low", 2:"Trop_high", 3:"Sub-trop_warm", 4:"Sub-trop_mod_cool", 5:"Sub-trop_cool", 
                                6:"Temp_mod", 7:"Temp_cool+Bor+Arctic"}, errors="raise")
mdum_soil = mdum_soil.rename(columns={1:"S1_very_steep", 2:"S2_hydro_soil", 3:"S3_no-slight_lim", 4:"S4_moderate_lim", 
                        5:"S5_severe_lim", 6:"L1_irr"}, errors="raise")
#merge the two dummy dataframes with the rest of the variables
dmaize_d_elim = pd.concat([dm0_elim, mdum_mst, mdum_thz, mdum_soil], axis='columns')
#drop the original mst and thz colums as well as one column of each dummy (this value will be encoded by 0 in all columns)
if(IRRIGATED):
    dmaize_dum_elim = dmaize_d_elim.drop(['270+days','Temp_cool+Bor+Arctic', 'L1_irr'], axis='columns')
else:
    dmaize_dum_elim = dmaize_d_elim.drop(['270+days','Temp_cool+Bor+Arctic'], axis='columns')

#select a random sample of 20% from the dataset to set aside for later validation
#random_state argument ensures that the same sample is returned each time the code is run
dmaize_val_elim = dmaize_dum_elim.sample(frac=0.2, random_state=2705) #RAW
#drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dmaize_fit_elim = dmaize_dum_elim.drop(dmaize_val_elim.index) #RAW
##################Collinearity################################

#caution takes up a lot of memory
#sb.pairplot(dmaize_fit_elim)

#extract lat, lon, area and yield from the fit dataset to test the correlations among the
#independent variables
if(IRRIGATED):
    dmaize_cor_elim = dmaize_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                         'irrigation_rel','thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
else:
    dmaize_cor_elim = dmaize_fit_elim.drop(['lat', 'lon', 'area', 'Y',
                                        'n_fertilizer', 'n_manure', 'n_man_prod',
                                        'thz_class',
                                        'mst_class', 'soil_class'], axis='columns')
#calculate spearman (rank transformed) correlation coeficcients between the 
#independent variables and save the values in a dataframe
sp_m = dmaize_cor_elim.corr(method='spearman')
#very noticable correlations among the fertilizer variables (as expected)
#interestingly, also very high correlations between irrigation and fertilizer variables

############Variance inflation factor##########################

Xm = add_constant(dmaize_cor_elim)
pd.Series([variance_inflation_factor(Xm.values, i) 
               for i in range(Xm.shape[1])], 
              index=Xm.columns)

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
#              C(mst_class) + C(soil_class) ', data=dmaize_fit_elim)
#Gamma distribution
if(IRRIGATED):
    # m_mod_elimg = smf.glm(formula='Y ~ n_total + mechanized + \
    #           C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_elim, 
    #           family=sm.families.Gamma(link=sm.families.links.log))
    m_mod_elimg = smf.glm(formula='Y ~ n_total + mechanized + irrigation_rel + p_fertilizer + pesticides_H + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_elim, 
               family=sm.families.Gamma(link=sm.families.links.log))

else:
    m_mod_elimg = smf.glm(formula='Y ~ n_total + p_fertilizer + pesticides_H + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)', data=dmaize_fit_elim, 
              family=sm.families.Gamma(link=sm.families.links.log))

#Nullmodel
m_mod_elim0 = smf.glm(formula='Y ~ 1', data=dmaize_fit_elim, family=sm.families.Gamma(link=sm.families.links.log))
#Fit models
#m_fit_elimn = m_mod_elimn.fit()
m_fit_elimg = m_mod_elimg.fit()
m_fit_elim0 = m_mod_elim0.fit()
#print results
#print(m_fit_elimn.summary())
#LogLik: -2547000; AIC: 5094000; BIC: 5094000
print('m_fit_elimg.summary()')
print(m_fit_elimg.summary())
print('m_fit_elim0.summary()')
print(m_fit_elim0.summary())


###########Fit statistics#############
#calculate pseudo R² for the Gamma distribution
# m_pseudoR_elim = 1-(121800/196880) #0.38135
#m_pseudoR_elim = 1-(53432/102030) # 0.4763 without the cells above 100 ha
# print('m_pseudoR_elim')
# print(m_pseudoR_elim)

#calculate AIC and BIC for Gamma
m_aic = m_fit_elimg.aic 
m_bic = m_fit_elimg.bic_llf
#LogLik: -2479200; AIC: 4958456; BIC: 4958677 (-3305574: this doesn't track)

########Validation against the validation dataset########

#select the independent variables from the val dataset
if(IRRIGATED):
    independent_columns=['p_fertilizer','n_total','pesticides_H','irrigation_rel','mechanized','thz_class','mst_class','soil_class']
else:
    independent_columns=['p_fertilizer','n_total','pesticides_H','mechanized','thz_class','mst_class','soil_class']

independent_column_indices = []
for i in np.arange(0,len(dmaize_val_elim.columns)):
    m_val_elim = dmaize_val_elim.iloc[:,[i]]
    column = dmaize_val_elim.columns[i]
    if(column in independent_columns):
        independent_column_indices.append(i)


m_val_elim = dmaize_val_elim.iloc[:,independent_column_indices]

#fit the model against the validation data
#pred_elim = m_fit_elimn.predict(m_val_elim)
m_pred_elimg = m_fit_elimg.predict(m_val_elim)


# print(dmaize_val_elim[])
# print(m_pred_elimg)
# quit()

# ################### Change Resolution ##################
print("r2_score(dmaize_val_elim['Y'], m_pred_elimg)") #0.3572
print(r2_score(dmaize_val_elim['Y'], m_pred_elimg)) #0.3572

r2_scores = []
five_minute = 5/60 #degrees
for scale in np.arange(1,250,5):
    step = five_minute*scale # degrees 
    to_bin = lambda x: np.floor(x / step) * step
    dmaize_val_elim["latbin"] = dmaize_val_elim['lat'].map(to_bin)
    dmaize_val_elim["lonbin"] = dmaize_val_elim['lon'].map(to_bin)
    groups = dmaize_val_elim.groupby(["latbin", "lonbin"])

    # print('by area')
    dmaize_val_elim['predicted'] = m_pred_elimg
    dmaize_val_elim['sumproduct']=dmaize_val_elim['Y']*dmaize_val_elim['area']
    dmaize_val_elim['sumproduct_pred']=dmaize_val_elim['predicted']*dmaize_val_elim['area']

    # dmaize_val_elim.assign(net_area=dmaize_val_elim['area']).groupby(["latbin", "lonbin"]).net_area.sum()


    newlist = pd.DataFrame({})
    newlist['area']=dmaize_val_elim.groupby(["latbin", "lonbin"]).area.sum()
    newlist['sumproduct']=dmaize_val_elim.groupby(["latbin", "lonbin"]).sumproduct.sum()
    newlist['sumproduct_pred']=dmaize_val_elim.groupby(["latbin", "lonbin"]).sumproduct_pred.sum()
    newlist['mean']= newlist['sumproduct'] / newlist['area']
    newlist['mean_pred']= newlist['sumproduct_pred'] / newlist['area']

    # print("len(newlist['mean'])")
    # print("len(newlist['mean_pred'])")

    # print(len(newlist['mean']))
    # print(len(newlist['mean_pred']))

    del dmaize_val_elim['sumproduct_pred']
    del dmaize_val_elim['sumproduct']
    del dmaize_val_elim['predicted']



    #calculate the R² scores
    #r2_score(dmaize_val_elim['Y'], pred_elim) #0.3711
    # print("r2_score(newlist['mean'], newlist['mean_pred'])") #0.3572
    # print(r2_score(newlist['mean'],newlist['mean_pred'])) #0.3572
    r2_scores.append(r2_score(newlist['mean'],newlist['mean_pred']))
plt.figure()
x = np.linspace(5/60,len(r2_scores)*5/60,len(r2_scores))
plt.scatter(x,r2_scores)
plt.title(CROP + ' Variable Resolution Validation R^2')


# x = np.linspace(1,len(r2_scores),int(len(r2_scores)/5))
# plt.xticks(x, np.linspace(5/60,len(r2_scores)*5/60,int(len(r2_scores)/5)))
plt.xlabel('resolution, degrees')
plt.ylabel('R^2 between predicted and validation data')
plt.show()
# print("r2_score(newlist['mean'], newlist['mean_pred'])") #0.3572
# print(r2_score(newlist['mean'],newlist['mean_pred'])) #0.3572
#.49432 without cells below 100ha

'''
#plot the predicted against the observed values
plt.scatter(pred_elim, dmaize_val_elim['Y'])
plt.scatter(pred_elimg, dmaize_val_elim['Y'])
plt.scatter(dmaize_val_elim['n_total'], pred_elimg)
plt.scatter(dmaize_val_elim['n_total'], dmaize_val_elim['Y'])

#plot the histogram
plt.hist(pred_elimg, bins=50)
plt.title('Maize yield ha/kg')
plt.xlabel('yield kg/ha')
plt.ylabel('density')


plt.hist(dmaize_val_elim['Y'], bins=50)

an_elim = pd.concat([dmaize_val_elim, pred_elim, pred_elimg], axis='columns')
an_elim = an_elim.rename(columns={0:"pred_elim", 1:"pred_elimg"}, errors="raise")
sb.lmplot(x='pred_elimg', y='Y', data=an_elim)
'''



##########RESIDUALS for the Gamma distribution#############


#select the independent variables from the fit dataset
m_fit_elim = dmaize_fit_elim.iloc[:,independent_column_indices]

#get the influence of the GLM model
m_stat_elimg = m_fit_elimg.get_influence()
#print(m_stat_elimg.summary_table()), there seems to be too much data

#store cook's distance in a variable
m_elimg_cook = pd.Series(m_stat_elimg.cooks_distance[0]).transpose()
m_elimg_cook = m_elimg_cook.rename("Cooks_d", errors="raise")

#store the actual yield, the fitted values on response and link scale, 
#the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
#the absolute value of the resp and the sqrt of the stud residuals in a dataframe
#reset the index but keep the old one as a column in order to combine the dataframe
#with Cook's distance
m_data_infl = { 'Yield': dmaize_fit_elim['Y'],
                'GLM_fitted': m_fit_elimg.fittedvalues,
               'Fitted_link': m_fit_elimg.predict(m_fit_elim, linear=True),
               'resid_pear': m_fit_elimg.resid_pearson, 
               'resid_stud' : m_stat_elimg.resid_studentized,
               'resid_resp_abs' : np.abs(m_fit_elimg.resid_response),
               'resid_stud_sqrt' : np.sqrt(np.abs(m_stat_elimg.resid_studentized)),
               'hat_matrix':m_stat_elimg.hat_matrix_diag}
m_elimg_infl = pd.DataFrame(data=m_data_infl).reset_index()
m_elimg_infl = pd.concat([m_elimg_infl, m_elimg_cook], axis='columns')


#take a sample of the influence dataframe to plot the lowess line
m_elimg_infl_sample = m_elimg_infl.sample(frac=0.2, random_state=3163)



##########Residual Plot############

#########Studentized residuals vs. fitted values on link scale######

# plot_elimg = plt.figure(4)
# plot_elimg.set_figheight(8)
# plot_elimg.set_figwidth(12)
# plt.scatter('Fitted_link', 'resid_stud', data=m_elimg_infl)
# plot_elimg.axes[0].set_title('Studentized Residuals vs Fitted on link scale')
# plot_elimg.axes[0].set_xlabel('Fitted values on link scale')
# plot_elimg.axes[0].set_ylabel('Studentized Residuals')
# plt.show()

#########Response residuals vs. fitted values on response scale#######
# plot_elimg = plt.figure(4)
# plot_elimg.set_figheight(8)
# plot_elimg.set_figwidth(12)


# plot_elimg.axes[0] = sb.residplot('GLM_fitted', 'Yield', data=m_elimg_infl, 
#                           #lowess=True, 
#                           scatter_kws={'alpha': 0.5}, 
#                           line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

# plot_elimg.axes[0].set_title('Residuals vs Fitted')
# plot_elimg.axes[0].set_xlabel('Fitted values')
# plot_elimg.axes[0].set_ylabel('Residuals')

# # annotations
# abs_resid = m_elimg_infl['resid_resp_abs'].sort_values(ascending=False)
# abs_resid_top_3 = abs_resid[:3]

# for i in abs_resid_top_3.index:
#     plot_elimg.axes[0].annotate(i, 
#                                xy=(m_elimg_infl['GLM_fitted'][i], 
#                                    m_elimg_infl['resid_resp_abs'][i]))

# plt.show()
###############QQ-Plot########################

# plt.figure()
# QQ = ProbPlot(m_elimg_infl['resid_stud'])
# plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

# plot_lm_2.set_figheight(8)
# plot_lm_2.set_figwidth(12)

# plot_lm_2.axes[0].set_title('Normal Q-Q')
# plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
# plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

# # annotations
# abs_norm_resid = np.flip(np.argsort(np.abs(m_elimg_infl['resid_stud'])), 0)
# abs_norm_resid_top_3 = abs_norm_resid[:3]

# for r, i in enumerate(abs_norm_resid_top_3):
#     plot_lm_2.axes[0].annotate(i, 
#                                xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
#                                    m_elimg_infl['resid_stud'][i]));
# plt.show()

############Cook's distance plot##########

#############Cook's distance vs. no of observation######
# plt.figure()
# #sort cook's distance value to get the value for the largest distance####
# cook_sort = m_elimg_cook.sort_values(ascending=False)
# #select all Cook's distance values which are greater than 4/n (n=number of datapoints)
# cook_infl = m_elimg_cook.loc[m_elimg_cook > (4/273772)].sort_values(ascending=False)

# #barplot for values with the strongest influence (=largest Cook's distance)
# #because running the function on all values takes a little longer
# plt.bar(cook_infl.index, cook_infl)
# plt.ylim(0, 0.01)

# #plots for largest 3 cook values, the ones greater than 4/n and all distance values
# plt.scatter(cook_infl.index[0:3], cook_infl[0:3])
# plt.scatter(cook_infl.index, cook_infl)
# plt.scatter(m_elimg_cook.index, m_elimg_cook)
# plt.ylim(0, 0.01)
# plt.show()
############Studentized Residuals vs. Leverage w. Cook's distance line#####

# plot_lm_4 = plt.figure(4)
# plot_lm_4.set_figheight(8)
# plot_lm_4.set_figwidth(12)

# plt.scatter(m_elimg_infl['hat_matrix'], m_elimg_infl['resid_stud'], alpha=0.5)
# sb.regplot(m_elimg_infl['hat_matrix'], m_elimg_infl['resid_stud'], 
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
# leverage_top_3 = np.flip(np.argsort(m_elimg_infl["Cooks_d"]), 0)[:3]

# for i in leverage_top_3:
#     plot_elimg.axes[0].annotate(i, 
#                                xy=(m_elimg_infl['hat_matrix'][i], 
#                                    m_elimg_infl['resid_stud'][i]))

# shenanigans for cook's distance contours
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(m_fit_elimg.params) # number of model parameters

# graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
#       np.linspace(0.001, 0.200, 50), 
#       'Cook\'s distance') # 0.5 line
# graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
#       np.linspace(0.001, 0.200, 50)) # 1 line
# plt.legend(loc='upper right');
# plt.show()

    

'''
#index of leverage top 3 is not the index of the datapoints, therefore I print
#the m_elimg_infl rows at this index because it contains the old index as a column
for i in leverage_top_3:
    print(m_elimg_infl.iloc[i])

sm.graphics.plot_regress_exog(m_fit_elimg, 'n_total')
plt.show()


'''


#########################################################################
################Loss of Industry Modelling###############################
#########################################################################

####################Data Prepping########################################

#take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
#and available manure correctly
LoI_melim = dm0_raw

LoI_melim['mechanized'] = LoI_melim['mechanized'].replace(-9,np.nan)
LoI_melim['pesticides_H'] = LoI_melim['pesticides_H'].replace(-9,np.nan)

############ Mechanised ##########################

#set mechanization to 0 in year 2, due to fuel estimations it could be kept the 
#same for 1 year
LoI_melim['mechanized_y2'] = LoI_melim['mechanized'].replace(1,0)

############ N fertilizer #########################

mn_drop= LoI_melim[((LoI_melim['mechanized'].isna())|(LoI_melim['pesticides_H'].isna()))
                & (LoI_melim['n_fertilizer']<0)].index
LoI_melim_pn = LoI_melim.drop(mn_drop)

#replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_melim_pn.loc[LoI_melim_pn['n_fertilizer'] < 0, 'n_fertilizer'] = np.nan #only 2304 left, so ffill 
LoI_melim_pn.loc[LoI_melim_pn['p_fertilizer'] < 0, 'p_fertilizer'] = np.nan
LoI_melim_pn[['n_fertilizer','p_fertilizer']] = LoI_melim_pn[['n_fertilizer','p_fertilizer']].fillna(method='ffill')
#replace no data values in n_total with the sum of the newly filled n_fertilizer and the
#n_manure values
LoI_melim_pn.loc[LoI_melim_pn['n_total'] < 0, 'n_total'] = LoI_melim_pn['n_fertilizer'] + LoI_melim_pn['n_manure']

#drop the nonsense values (99.9th percentile) in the n and p fertilizer columns
LoI_melim_pn = LoI_melim_pn.loc[LoI_melim_pn['n_fertilizer'] < dm0_qt.iloc[12,4]]#~180
LoI_melim_pn = LoI_melim_pn.loc[LoI_melim_pn['p_fertilizer'] < dm0_qt.iloc[12,5]] #~34

#in year 1, there will probably be a slight surplus of N (production>application)
#calculate kg N applied per cell
LoI_melim_pn['n_kg'] = LoI_melim_pn['n_fertilizer']*LoI_melim_pn['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_pn['n_ffrac'] = LoI_melim_pn['n_kg']/(LoI_melim_pn['n_kg'].sum())

#calculate the fraction of total N applied to maize fields on the total N applied
#divide total of maize N by 1000000 to get from kg to thousand t
m_nfert_frac = (LoI_melim_pn['n_kg'].sum())/1000000/118763
#calculate the new total for N maize in year one based on the N total surplus
m_ntot_new = m_nfert_frac * 14477 * 1000000

#calculate the new value of N application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melim_pn['n_fert_y1'] = (m_ntot_new * LoI_melim_pn['n_ffrac']) / LoI_melim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melim_pn['n_fert_y2'] = 0
#LoI_melim_pn.loc[LoI_melim_pn['n_fert_y2'] > 0, 'n_fert_y1'] = 0

############## P Fertilizer #####################

#in year 1, there will probably be a slight surplus of P (production>application)
#calculate kg p applied per cell
LoI_melim_pn['p_kg'] = LoI_melim_pn['p_fertilizer']*LoI_melim_pn['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_pn['p_ffrac'] = LoI_melim_pn['p_kg']/(LoI_melim_pn['p_kg'].sum())

#calculate the fraction of total P applied to maize fields on the total P applied to cropland
#divide total of maize P by 1000000 to get from kg to thousand t
m_pfert_frac = (LoI_melim_pn['p_kg'].sum())/1000000/45858
#calculate the new total for P maize in year one based on the P total surplus
m_ptot_new = m_pfert_frac * 4142 * 1000000

#calculate the new value of P application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melim_pn['p_fert_y1'] = (m_ptot_new * LoI_melim_pn['p_ffrac']) / LoI_melim_pn['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melim_pn['p_fert_y2'] = 0

############# N Manure ###################

#drop the rows containing nonsense values (99th percentile) in the manure column
LoI_melim_man = LoI_melim.loc[LoI_melim['n_manure'] < dm0_qt.iloc[10,6]] #~11

#calculate kg N applied per cell: 1,018,425,976.75 kg total
LoI_melim_man['man_kg'] = LoI_melim_man['n_manure']*LoI_melim_man['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_man['n_mfrac'] = LoI_melim_man['man_kg']/(LoI_melim_man['man_kg'].sum())

#calculate the fraction of total N applied to maize fields of the total N applied to cropland
#divide total of maize N by 1000000 to get from kg to thousand t
m_nman_frac = (LoI_melim_man['man_kg'].sum())/1000000/24000

#calculate animal labor demand by dividing the area in a cell by the area a cow
#can be assumed to work
LoI_melim_man['labor'] = LoI_melim_man['area']/5 #current value (7.4) is taken from Dave's paper
#might be quite inaccurate considering the information I have from the farmer
#I chose 5 now just because I don't believe 7.4 is correct

#calculate mean excretion rate of each cow in one year: cattle supplied ~ 43.7% of 131000 thousand t
#manure production in 2014, there were ~ 1.008.570.000(Statista)/1.439.413.930(FAOSTAT) 
#heads of cattle in 2014
cow_excr = 131000000000*0.437/1439413930

#calculate available manure based on cow labor demand: 1,278,868,812.065 kg
m_man_av = cow_excr * LoI_melim_man['labor'].sum()
#more manure avialable then currently applied, but that is good as N from mineral
#fertilizer will be missing

#calculate the new value of man application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melim_man['man_fert'] = (m_man_av * LoI_melim_man['n_mfrac']) / LoI_melim_man['area']


########### N total ######################

LoI_melim['N_toty1'] = LoI_melim_pn['n_fert_y1'] + LoI_melim_man['man_fert']
#multiply area with a factor which accounts for the reduction of farmed area due to
#longer/different crop rotations being necessary to induce enough nitrogen and
#recovery times against pests in the rotation
LoI_melim['area_LoI'] = LoI_melim['area']*(2/3) #value is just a placeholder
#maybe this is not the way, because it's something the calculation doesn't account for:
# if less pesticides are used, the yield will go down accordingly without considering rotation
#maybe it accounts for it implicitly, though: farms with zero to low pesticide use
#probably have different crop rotations

############## Pesticides #####################

LoI_melimp = LoI_melim.loc[LoI_melim['pesticides_H'].notna()]
LoI_melimp = LoI_melimp.loc[LoI_melimp['pesticides_H'] < dm0_qt.iloc[12,9]]#~11
#in year 1, there will probably be a slight surplus of Pesticides (production>application)
#calculate kg p applied per cell
LoI_melimp['pest_kg'] = LoI_melimp['pesticides_H']*LoI_melimp['area']
#calculate the fraction of the total N applied to maize fields for each cell
LoI_melimp['pest_frac'] = LoI_melimp['pest_kg']/(LoI_melimp['pest_kg'].sum())

#calculate the fraction of total pesticides applied to maize fields on the total pesticides applied to cropland
#divide total of maize pesticides by 1000 to get from kg to t
m_pest_frac = (LoI_melimp['pest_kg'].sum())/1000/4190985

#due to missing reasonable data on the pesticide surplus, it is assumed that the
#surplus is in the same range as for P and N fertilizer
frac_pest = ((14477/118763) + (4142/45858))/2
#calculate the new total for pesticides maize in year one based on the pesticides total surplus
m_pestot_new = m_pest_frac * (4190985 * frac_pest) * 1000

#calculate the new value of pesticides application rate in kg per ha per cell, assuming
#the distribution remains the same as before the catastrophe
LoI_melimp['pest_y1'] = (m_pestot_new * LoI_melimp['pest_frac']) / LoI_melimp['area']

#in year 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melimp['pest_y2'] = 0


########## Irrigation ####################

#in LoI it is assumed that only irrigation which is not reliant on electricity
#can still be maintained
#calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
#(fraction of irrigated cropland reliant on electricity)
if(IRRIGATED):
    LoI_melim['irr_LoI'] = LoI_melim['irrigation_tot'] * (1- LoI_melim['irrigation_rel'])

###########Combine the different dataframes and drop rows with missing values#########

LoI_melim = pd.concat([LoI_melim, LoI_melim_pn['n_fert_y1'], LoI_melim_pn['n_fert_y2'],
                       LoI_melim_pn['p_fert_y1'], LoI_melim_pn['p_fert_y2'],
                       LoI_melim_man['man_fert'], LoI_melimp['pest_y1'], 
                       LoI_melimp['pest_y2']], axis='columns')

#Handle the data by eliminating the rows without data:
LoI_melim = LoI_melim.dropna()

#Handle outliers by eliminating all points above the 99.9th percentile
#I delete the points because the aim of this model is to predict well in the lower yields
#dm0_qt = dm0_elim.quantile([.1, .25, .5, .75, .8,.85, .87, .9, .95,.975, .99,.995, .999,.9999])
#dm0_qt.reset_index(inplace=True, drop=True)
LoI_melim = LoI_melim.loc[LoI_melim['Y'] < dm0_qt.iloc[12,3]] #~12500
#dm0_elim = dm0_elim.loc[dm0_elim['n_man_prod'] < dm0_qt.iloc[12,7]] #~44
LoI_melim = LoI_melim.loc[LoI_melim['n_total'] < dm0_qt.iloc[12,8]] #~195

# quit()

#########################Prediction of LoI yields#########################

################## Year 1 ##################

#select the rows from LoI_melim which contain the independent variables for year 1

# independent_columns=['thz_class','mst_class','soil_class','120-180days','225-270days','Sub-trop_warm','Temp_mod']
independent_columns = ['mechanized', 'thz_class', 'mst_class', 'soil_class',
    'N_toty1','irr_LoI', 'p_fert_y1', 'pest_y1']

independent_column_indices = []
for i in np.arange(0,len(LoI_melim.columns)):
    m_val_elim = LoI_melim.iloc[:,[i]]
    column = LoI_melim.columns[i]
    if(column in independent_columns):
        independent_column_indices.append(i)


LoI_m_year1 = LoI_melim.iloc[:,independent_column_indices]

#reorder the columns according to the order in dm0_elim
if(IRRIGATED):
    LoI_m_year1 = LoI_m_year1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                           'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_m_year1 = LoI_m_year1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                          'pest_y1':"pesticides_H",
                                          'irr_LoI':"irrigation_tot"}, errors="raise")
else:
    LoI_m_year1 = LoI_m_year1[['p_fert_y1', 'N_toty1', 'pest_y1', 'mechanized', 
                       'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_m_year1 = LoI_m_year1.rename(columns={'p_fert_y1':"p_fertilizer", 'N_toty1':"n_total", 
                                          'pest_y1':"pesticides_H"}, errors="raise")
#predict the yield for year 1 using the gamma GLM
m_yield_y1 = m_fit_elimg.predict(LoI_m_year1)
#calculate the change rate from actual yield to the predicted yield
m_y1_change = (m_yield_y1-maize_kgha)/maize_kgha

#calculate statistics for yield and change rate

#yield
mmean_y1_weigh = round(np.average(m_yield_y1, weights=LoI_melim['area']),2) #3832.02kg/ha
mmax_y1 = m_yield_y1.max() #10002.44 kg/ha
mmin_y1 = m_yield_y1.min() #691.74 kg/ha

#change rate
#mmean_y1c_weigh = round(np.average(m_y1_change, weights=maize_yield['growArea']),2)
mmax_y1c = m_y1_change.max() # +105.997 (~+10600%)
mmin_y1c = m_y1_change.min() #-0.94897 (~-95%)

################## Year 2 ##################

#select the rows from LoI_melim which contain the independent variables for year 2
# independent_columns=['thz_class','mst_class','soil_class','LGP<120days','225-270days','Sub-trop_mod_cool','Sub-trop_cool','S1_very_steep']
independent_columns=['thz_class', 'mst_class', 'soil_class', 'mechanized_y2', 'irr_LoI',
       'p_fert_y2', 'man_fert', 'pest_y2']

independent_column_indices = []
for i in np.arange(0,len(LoI_melim.columns)):
    m_val_elim = LoI_melim.iloc[:,[i]]
    column = LoI_melim.columns[i]
    if(column in independent_columns):
        independent_column_indices.append(i)

LoI_m_year2 = LoI_melim.iloc[:,independent_column_indices]#reorder the columns according to the order in dm0_elim
if(IRRIGATED):
    LoI_m_year2 = LoI_m_year2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 
                           'irr_LoI', 'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_m_year2 = LoI_m_year2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                          'pest_y2':"pesticides_H",'mechanized_y2':"mechanized",
                                          'irr_LoI':"irrigation_tot"}, errors="raise")
else:
    LoI_m_year2 = LoI_m_year2[['p_fert_y2', 'man_fert', 'pest_y2', 'mechanized_y2', 'thz_class', 'mst_class', 'soil_class']]
    #rename the columns according to the names used in the model formular
    LoI_m_year2 = LoI_m_year2.rename(columns={'p_fert_y2':"p_fertilizer", 'man_fert':"n_total", 
                                          'pest_y2':"pesticides_H",'mechanized_y2':"mechanized"}, errors="raise")    

#predict the yield for year 2 using the gamma GLM
m_yield_y2 = m_fit_elimg.predict(LoI_m_year2)
#calculate the change from actual yield to the predicted yield
m_y2_change = (m_yield_y2-maize_kgha)/maize_kgha

#calculate statistics for yield and change rate

#yield
mmean_y2_weigh = round(np.average(m_yield_y2, weights=LoI_melim['area']),2) #2792.08kg/ha
mmax_y2 = m_yield_y2.max() #6551.74kg/ha
mmin_y2 = m_yield_y2.min() #689.79kg/ha

#change rate
mmean_y2c = m_y2_change.mean() #0.1198 (~+12%)
mmax_y2c = m_y2_change.max() #70.087 (~+7000%)
mmin_y2c = m_y2_change.min() #-0.9503 (~-95%)

#combine both yields and change rates with the latitude and longitude values
LoI_maize = pd.concat([maize_yield['lats'], maize_yield['lons'], m_yield_y1,
                m_y1_change, m_yield_y2, m_y2_change], axis='columns')
#save the dataframe in a csv
LoI_maize.to_csv(params.geopandasDataDir + "LoI"+CROP+"Yield"+postfix+"Filtered.csv")
print('saved')