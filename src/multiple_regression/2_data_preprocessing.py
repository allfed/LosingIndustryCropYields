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

from utilities import params
from utilities import stat_ut
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

params.importAll()

crops = {"Corn", "Rice", "Soybean", "Wheat"}



'''
Load raw data and eliminate all rows with an area below 100 ha
'''
data_raw, data_step1 = {}, {}
for crop in crops:
    data_name = '{}_raw'.format(crop)
    data_raw[crop] = pd.read_pickle(params.geopandasDataDir + data_name + '.pkl', compression='zip')
    #crop_name = '{}_clean'.format(crop)
    data_step1[crop] = data_raw[crop].loc[data_raw[crop]['area']>100]
    
print('Done reading crop data and eliminate all rows below 100 ha')

'''
Calculate area of raw datasets for specified columns to use them in LoI_scenario_data.py
'''
def calculate_area(data, columns):
    dict_area = {}
    for col in columns:
        dict_area[col] = data['area'].loc[data[col]>=0].sum().astype('int')
    return dict_area

#specify columns and apply function for all crops
columns = ['n_fertilizer', 'p_fertilizer', 'pesticides']
area_data, area_stat = {}, {}
for crop in crops:
    #calculate_area selects valid rows based on values >=0 so the no data value
    #of the dataframe is set to a negative number
    area_data[crop] = data_raw[crop].fillna(-9)
    area_stat[crop] = calculate_area(area_data[crop], columns)
    
#area_data['Corn'] = data_raw['Corn'].fillna(-9)
#data_raw['Corn']= data_raw['Corn']['continents'].astype('int8')
#data_raw['Corn'].dtypes
#convert dict to dataframe and save to csv
area_col = pd.DataFrame(area_stat)
area_col.to_csv(params.geopandasDataDir + 'Raw_Column_Area.csv') 

print('Done calculating total crop area and saving it to csv')  

'''
I still have a couple of problems in the following for loop to calculate min, max, weighted mean and NaN count
of the raw dataset:
    1. variable names are not the best yet: have to change test and desc_stat name
    2.new variable for values that concern the whole dataset: total NaN rows?, total crop area raw and clean,
    number of rows raw and clean
    3. write function for one block of descriptive statistics.
    4. turn dict into multiindexed dataframe? Already in the beginning or in the end when exporting it?

#numerical = ['area', 'Yield', 'n_fertilizer', 'p_fertilizer', 'n_manure', 'n_total', 'pesticides', 'irrigation_tot']

test, desc_stat = {}, {}
for crop in crops:
    data_name = '{}_raw'.format(crop)
    cat = {'mechanized':1, 'thz_class':2, 'mst_class':3, 'soil_class':4}
    test[data_name] = data_raw[crop].drop(['lat', 'lon'], axis='columns')\
                                                .replace([-99989.9959564209, -9], np.nan) # I think one time np.nan without brackets suffices
    desc_stat[data_name] = pd.DataFrame(test[data_name].isna().sum(), columns=['NaN Count'])
    desc_stat[data_name]['0 count'] = (test[data_name] == 0).sum()
    desc_stat[data_name]['Max'] = test[data_name].max()
    desc_stat[data_name]['Min'] = test[data_name].min()
    desc_stat[data_name]['Weighted Mean_Mode'] = test[data_name].apply(lambda x: stat_ut.weighted_average(\
                                                                        x, weights=test[data_name]['area'], dropna=True))
    #desc_stat[data_name]['Weighted Mean_Mode'] = np.average(test[data_name], weights=test[data_name]['area'], axis=0)
    desc_stat[data_name].loc[cat, 'Weighted Mean_Mode'] = test[data_name].loc[:,cat].apply(lambda x: \
                                                          stat_ut.weighted_mode(x, weights=test[data_name]['area'], dropna=True))
    #desc_stat[data_name].loc[cat, 'Weighted Mean_Mode'] = np.transpose(test[data_name].loc[:,cat].mode()).values.tolist()

    crop_name = '{}_clean'.format(crop)
    test[crop_name] = data_step1[crop].drop(['lat', 'lon'], axis='columns')\
                                            .replace([-99989.9959564209, -9], np.nan)
    desc_stat[crop_name] = pd.DataFrame(test[crop_name].isna().sum(), columns=['NaN Count'])
    desc_stat[crop_name]['0 count'] = (test[crop_name] == 0).sum()
    desc_stat[crop_name]['Max'] = test[crop_name].max()
    desc_stat[crop_name]['Min'] = test[crop_name].min()
    desc_stat[crop_name]['Weighted Mean_Mode'] = test[crop_name].apply(lambda x: stat_ut.weighted_average(\
                                                                        x, weights=test[crop_name]['area'], dropna=True))
    desc_stat[crop_name].loc[cat, 'Weighted Mean_Mode'] = test[crop_name].loc[:,cat].apply(lambda x: \
                                                          stat_ut.weighted_mode(x, weights=test[crop_name]['area'], dropna=True))
    #desc_stat[crop_name]['Weighted Mean_Mode'] = np.average(test[crop_name], weights=test[crop_name]['area'], axis=0)
    #desc_stat[crop_name].loc[cat, 'Weighted Mean_Mode'] = np.transpose(test[crop_name].loc[:,cat].mode()).values.tolist()
'''

'''
Combine some of the levels of AEZ classes, fill or eliminate missing data
in soil, n+p fertilizer, n_total, pesticides, mechanized and irrigation_rel
'''
#function to fill classes with unreasonable values (for cropland) ins soil class
#and combine some of the levels of the thz & mst class to ensure that levels with
#few datapoints don't introduce imbalance into the data
def clean_aez(data):
    data_aez = data.copy()
    #replace 0s, 7s & 8as in the soil class with NaN values so they can be handled with the .fillna method
    data_aez['soil_class'] = data_aez['soil_class'].replace([0, 7, 8], np.nan)
    #fill in the NaN vlaues in the dataset with a forward filling method
    #(replacing NaN with the value in the cell before)
    data_aez = data_aez.fillna(downcast='infer', method='ffill')
    #replace 8,9 & 10 with 7 in the thz class to combine all 3 classes into 1 Temp,cool-Arctic class
    #repalce 2 with 1 and 7 with 6 in the mst class to compile them into 1 joined class each
    data_aez['thz_class'] = data_aez['thz_class'].replace([8, 9, 10], 7)
    data_aez['mst_class'] = data_aez['mst_class'].replace({2:1, 7:6})
    return data_aez

#function to fill missing values in the n_fertilizer, p_fertilizer and n_total columns
def clean_fertilizer(data):
    data_fertilizer = data.copy()
    no_data_value = data_fertilizer['n_fertilizer'].min()
    #replace remaining no data values in fertilizer datasets with NaN, then fill them
    data_fertilizer = data_fertilizer.replace({'n_fertilizer':no_data_value, 'p_fertilizer':no_data_value}, np.nan)
    data_fertilizer = data_fertilizer.fillna(method='ffill')
    #replace NaN values in n_total with sum of the newly filled n_fertilizer and n_manure values
    data_fertilizer.loc[data_fertilizer['n_total'] < 0, 'n_total'] = data_fertilizer['n_fertilizer'] + data_fertilizer['n_manure']
    return data_fertilizer

#for each crop: remove rows with missing data in pesticides and mechanized columns, replace missing values in
#irrigation_rel with 0 and apply the above defined functions
data_step2 = {}
for crop in crops:
    #replace NaN values with -9 so that fillna can be used on the entire dataframe in the next step
    data_step2[crop] = data_step1[crop].replace({'pesticides':np.nan, 'irrigation_rel':np.nan}, {'pesticides':-9, 'irrigation_rel':0})
    #fill NaN values in soil_class and combine categories in thz & mst class
    data_step2[crop] = clean_aez(data_step2[crop])
    #fill NaN values in n & p fertilizer and n total columns
    data_step2[crop] = clean_fertilizer(data_step2[crop])
    #Eliminate the rows without data in the pesticides and mechanized columns
    data_step2[crop] = data_step2[crop].query('pesticides > -9 and mechanized > -9')
    
print('Done combining AEZ classes, fill or eliminate missing data')

'''
Calculate and eliminate outliers (values above the 99th/99.9th quantile) from the dataset
and extract the outliers to calculate outlier statistics in a later step
'''
#select the columns of the dataframe where outliers will be calculated
factors = ['Yield', 'n_fertilizer', 'p_fertilizer', 'n_manure', 'n_total', 'pesticides']
#calculate the 99.9th quantile for the columns specified above
quant = data_step2['Corn'][factors].quantile(.999)
#combine both into a dictionary
out_threshold = dict(zip(factors, quant))
#replace the value for n_manure with the 99th quantile
out_threshold['n_manure'] =  data_step2['Corn']['n_manure'].quantile(.99)

#function to remove all rows from a dataframe where the values of the specified
#columns surpass the specified threshold
def eliminate_outliers(data, factors, thresholds):
  data_outliers = data.loc[np.logical_and.reduce([data[factor]<thresholds[factor] for factor in factors])]
  return data_outliers

#function to extract all rows from a dataframe where the values of the specified
#columns surpass the specified threshold and combine the results with the area
#column in one dataframe (area is needed for weighted mean calculations in a later step)
def extract_outliers(data, factors, thresholds):
  outlier = {}
  for factor in factors:
    outlier[factor] = data[factor].loc[data[factor] >= thresholds[factor]]
  outliers = pd.DataFrame(outlier, columns=factors)
  results = pd.concat([outliers, data['area'][data['area'].index.isin(outliers.index)]], axis='columns')
  return results

#for each crop: apply the above functions and save the resulting data_clean dataframes to csv files
data_step3 = {}
outliers = {}
for crop in crops:
    data_step3[crop] = eliminate_outliers(data_step2[crop], factors, out_threshold)
    outliers[crop] = extract_outliers(data_step2[crop], factors, out_threshold)

print('Done extracting and eliminating outliers')  

'''
Replace No Data Values in continent column with corresponding continent value
'''

#Create lists with the continent values for the missing data points
fill_values_Corn = [4] * 11 + [6, 6, 4, 4, 4, 6, 1, 5, 1,
                   1, 1, 5, 1, 1, 1, 1, 1] + [2] * 39 
fill_values_Rice = [6] * 12 + [1] * 10 + [2] * 133
fill_values_Soybean = [4] * 3 + [2] * 9
fill_values_Wheat = [4, 6, 5, 5, 5, 5, 5, 1, 5, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2]

#store the lists in a dictionary
continents_NoData = {'fill_values_Corn': fill_values_Corn,
                     'fill_values_Rice': fill_values_Rice,
                     'fill_values_Soybean': fill_values_Soybean,
                     'fill_values_Wheat': fill_values_Wheat}

data_clean ={}
for crop in crops:
    fill_values = 'fill_values_{}'.format(crop)
    #for each crop, add the index of rows which are missing continent data to the dictionary
    continents_NoData[crop] = data_step3[crop].loc[data_step3[crop]['continents']==0].index
    #initialize data_clean by creating a copy of data_step3
    data_clean[crop] = data_step3[crop].copy()
    #set the cells where continent==0 to the values stored in the lists
    data_clean[crop].loc[continents_NoData[crop], 'continents'] = continents_NoData[fill_values]
    #save the clean dataset to csv
    data_clean[crop].to_csv(params.geopandasDataDir + crop + '_data.gzip', compression='gzip')


print('Done replacing no data values in the continent column and saving the clean dataset to file')  


'''
Overview Stats for each step and each crop

metrics = ['Tot_Area', 'Numb_Rows', 'Numb_Outliers']
steps = ['raw', 'step1', 'step2', 'clean']
overview_stats = {}
for crop, metric in crops, metrics:
    for step in steps:
        step_name = 'data_{}'.format(step)
'''
'''
Dummy-code the categorical variables to be able to assess multicollinearity
'''
#mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########

aez_classes = ['thz_class', 'mst_class', 'soil_class']
aez_names = {'TRC_2': "Trop_high", 'TRC_3': "Sub-trop_warm", 'TRC_4': "Sub-trop_mod_cool", 'TRC_5': "Sub-trop_cool", 'TRC_6': "Temp_mod",
             'TRC_7': "Temp_cool+Bor+Arctic", 'M_3': "120-180days", 'M_4': "180-225days", 'M_5': "225-270days", 'M_6': "270+days",
             'S_2': "S2_hydro_soil", 'S_3': "S3_no-slight_lim", 'S_4': "S4_moderate_lim", 'S_5': "S5_severe_lim", 'L_3': "L3_irr"}
dum = pd.get_dummies(data_clean['Corn'], prefix=['TRC', 'M', 'S'], columns=aez_classes, drop_first=True).rename(columns={'S_6':'L_3'})
data_dummy = pd.concat([dum.iloc[:,:12], data_clean['Corn'][aez_classes],dum.iloc[:,12:]], axis='columns').rename(columns=aez_names, errors='raise')
#dumdum = data_dummy.rename(columns=aez_names, errors='raise')

data_dummy = {}
#have to decide if I want to keep the TRC,M,S classification or if I want to rename them according to the aez_names(see above)
for crop in crops:
    dummies = pd.get_dummies(data_clean[crop], prefix=['TRC', 'M', 'S'], columns=aez_classes, drop_first=True).rename(columns={'S_6':'L_3'})
    data_dummy[crop] = pd.concat([dummies.iloc[:,:12], data_clean[crop][aez_classes],dummies.iloc[:,12:]], axis='columns')#.rename(columns=aez_names, errors='raise')
    #data_dummy[crop].to_csv(params.geopandasDataDir + crop + '_dummies.gzip', compression='gzip')


'''
''
#Generate Plots <- needs more detail
''
def plot_yield(data, crop, bound):
    param = stats.gamma.fit(data)
    x = np.linspace(0.01,
                bound, 100)
    pdf_fitted = stats.gamma.pdf(x, *param)
    # sets design aspects for the following plots
    matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
    matplotlib.style.use('ggplot')

    # plot wheat yield distribution histogram and fitted gamma distribution
    plt.hist(data, bins=50, density=True, color='darkkhaki')
    plt.plot(x, pdf_fitted, lw=2, label="Fitted Gamma distribution", color='navy')
    plt.title(crop + ' yield ha/kg')
    plt.xlabel('yield kg/ha')
    plt.ylabel('density')
    plt.xlim(right=bound)
    plt.legend()
    plt.show()
    
    #plt.savefig(params.figuresDir + 'my_plot.png')
    #plt.close()
    
def plot_yield(data, crops, bounds):
    data_name = data.keys
    crop_name = crops[crop]  
    param = stats.gamma.fit(data)
    x = np.linspace(0.01,
                bound, 100)
    pdf_fitted = stats.gamma.pdf(x, *param)
    # sets design aspects for the following plots
    matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
    matplotlib.style.use('ggplot')
    fig, axes = plt.subplots(2, 2)

    for d in data_name:
        
    # plot wheat yield distribution histogram and fitted gamma distribution
        plt.hist(data[d]['Yield'], bins=50, density=True, color='darkkhaki', ax=axes[0])
        plt.plot(x, pdf_fitted, lw=2, label="Fitted Gamma distribution", color='navy')
        plt.title(crop + ' yield ha/kg')
        plt.xlabel('yield kg/ha')
        plt.ylabel('density')
        plt.xlim(right=bounds[d])
        plt.legend()
        plt.show()
    
    #plt.savefig(params.figuresDir + 'my_plot.png')
    #plt.close()
    
plot_yield(crop_raw['Corn_raw']['Yield'], 'Corn', 20000)

#Boxplot of all the variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle('dw0_raw Boxplots for each variable')

sb.boxplot(ax=axes[0, 0], data=dw0_raw, x='n_fertilizer')
sb.boxplot(ax=axes[0, 1], data=dw0_raw, x='p_fertilizer')
sb.boxplot(ax=axes[0, 2], data=dw0_raw, x='n_manure')
sb.boxplot(ax=axes[1, 0], data=dw0_raw, x='n_total')
sb.boxplot(ax=axes[1, 1], data=dw0_raw, x='pesticides')
sb.boxplot(ax=axes[1, 2], data=dw0_raw, x='Y')

ax = sb.boxplot(x=dw0_raw["Y"], orient='v')
ax = sb.boxplot(x=dw0_raw["n_fertilizer"])
ax = sb.boxplot(x=dw0_raw["p_fertilizer"])
ax = sb.boxplot(x=dw0_raw["n_manure"])
ax = sb.boxplot(x=dw0_raw["n_total"])
ax = sb.boxplot(x=dw0_raw["pesticides_H"])
ax = sb.boxplot(x=crop_clean['Corn']["irrigation_tot"])
ax = sb.boxplot(x=crop_clean['Corn']["irrigation_rel"])
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
sb.boxplot(ax=axes[0, 0], data=crop_clean['Corn'], x='mechanized', y='Yield')
plt.xlim(0.5,2.5)
plt.ylim(0,20000)
sb.boxplot(ax=axes[0, 1], data=crop_clean['Corn'], x='thz_class', y='Yield')
plt.ylim(0,20000)
sb.boxplot(ax=axes[1, 0], data=crop_clean['Corn'], x='mst_class', y='Yield')
plt.ylim(0,20000)
sb.boxplot(ax=axes[1, 1], data=crop_clean['Corn'], x='soil_class', y='Yield')
plt.ylim(0,20000)
ax = sb.lineplot(x="n_fertilizer", y='Yield', data=crop_clean['Corn'])
plt.xlim(-0.5,4500)
ax = sb.relplot(x="irrigation_tot", y="Yield", data=crop_clean['Corn'], kind="scatter")
plt.xlim(-0.5,4500)
crop_clean['Corn'].max()
fig, ax = plt.subplots()
ax.bar(x="n_fertilizer", height='Yield', data=crop_clean['Corn'])
ax = sb.boxenplot(x="mechanized", y='Yield', data=crop_clean['Corn'])
plt.xlim(0.5,2.5)
plt.ylim(0,20000)
ax = sb.violinplot(x="thz_class", y='Yield', data=crop_clean['Corn'], inner="stick")
plt.ylim(0,20000)
ax = sb.boxplot(x="mst_class", y='Yield', data=crop_clean['Corn'])
plt.ylim(0,20000)
ax = sb.boxplot(x="soil_class", y='Yield', data=crop_clean['Corn'])
plt.ylim(0,20000)


''
#Check for multicollinearity by calculating the two-way correlations and the VIF
''

#extract lat, lon, area, yield, individual n columns, original climate class columns and irrigation for the LoI scenario
#from the fit dataset to test the correlations among the
#independent variables
dwheat_cor_elim = data_dummy[crop].drop(['lat', 'lon', 'area', 'Y',
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
'''

