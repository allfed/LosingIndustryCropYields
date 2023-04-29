# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:25:39 2022

@author: Jess
"""
import os
import src.utilities.params as params  # get file location and varname parameters for
import src.utilities.utilities as utilities  # get file location and varname parameters for
import src.utilities.stat_ut as stat_ut  # get file location and varname parameters for
import pandas as pd
from scipy import stats
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import d2_tweedie_score
from sklearn.metrics import mean_tweedie_deviance

params.importAll()
#######################################################################
########### Regression Calibration, Validation and Residuals###########
#######################################################################

crops = {"Corn", "Rice", "Soybean", "Wheat"}

"""
Load the clean dataset for each crop and split the data into a validation and a calibration dataset
"""

model_data, val_data, val_factors, fit_data = {}, {}, {}, {}
for crop in crops:
    model_data[crop] = pd.read_csv(
        params.modelDataDir + crop + "_data.gzip", index_col=0, compression="gzip"
    )
    # select a random sample of 20% from the dataset to set aside for later validation
    # random_state argument ensures that the same sample is returned each time the code is run
    val_data[crop] = model_data[crop].sample(frac=0.2, random_state=2705)
    #select the independent variables from the validation dataset
    val_factors[crop] = val_data[crop].iloc[:, [5, 7, 8, 9, 11, 12, 13, 14]]
    # drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
    fit_data[crop] = model_data[crop].drop(val_data[crop].index)

#load the world coordinates to later use them in creating the ascii output files    
coordinates = pd.read_pickle(params.LoIDataDir  + "Coordinates.pkl", compression="zip")

"""
Calibrate the Regression model, calculate and save fit statistics and validate the model
against the validation dataset
"""

#specify and fit the model for each crop, predict values for the validation dataset
model, fit, val = {}, {}, {}
for crop in crops:
    # specify the model for each crop with a gamma distribution and the independent factors total nitrogen application,
    #articificial phosphorus fertilizer application, total irrigation, pesticide application, temperature + 
    #moisture + soil class and a dummy indicating if the area is worked with agricultural machinery
    model[crop] = smf.glm(
        formula="Yield ~ n_total + p_fertilizer + irrigation_tot + mechanized + pesticides +  C(thz_class) + \
                  C(mst_class) + C(soil_class)",
              data=fit_data[crop],
              family=sm.families.Gamma(link=sm.families.links.log),
              )
    # Fit the models to the data
    fit[crop] = model[crop].fit()
    # let the models predict yield values for the validation data
    val[crop] = fit[crop].predict(val_factors[crop])

#calculate model parameters and statistics for each crop and combine them in a dataframe
model_results, model_stats, df_list = {}, {}, []
for crop in crops:
    #extract model parameters, their 95% Confidence Intervals and their p-values from the model,
    #calculate the odds ratios and combine all four values in a dataframe
    model_results[crop] = pd.DataFrame(fit[crop].params, columns=["Coefficients"])
    model_results[crop]['+-95%Confidence_Interval'] = fit[crop].bse * 2
    model_results[crop]['Odds_ratios'] = np.exp(fit[crop].params)
    model_results[crop]['p-value'] = fit[crop].pvalues
    #calculate McFadden's roh² and the Root Mean Gamma Deviance (RMGD) for 
    #the fitted values of the models and the values the models predict 
    #for the validation dataset
    #calculate the AIC and BIC for each model
    model_stats[crop] = [], []
    model_stats[crop][0].append(d2_tweedie_score(fit_data[crop]["Yield"], fit[crop].fittedvalues, power=2))
    model_stats[crop][0].append(np.sqrt(mean_tweedie_deviance(fit_data[crop]["Yield"], fit[crop].fittedvalues, power=2)))
    model_stats[crop][0].append(fit[crop].aic)
    model_stats[crop][0].append(fit[crop].bic_llf)
    model_stats[crop][1].append(d2_tweedie_score(val_data[crop]["Yield"], val[crop], power=2))
    model_stats[crop][1].append(np.sqrt(mean_tweedie_deviance(val_data[crop]["Yield"], val[crop], power=2)))
    model_stats[crop][1].extend([np.nan]*2)
    #combine the lists into a dataframe with a multiindex and create a list of dataframes
    #containing one dataframe for each crop
    df = pd.DataFrame(model_stats[crop], index=['Calibration', 'Validation'], columns=['McFaddens_roh', 'RootMeanGammaDeviance', 'AIC', 'BIC'])
    df.index = pd.MultiIndex.from_product([[crop], df.index])
    df_list.append(df)
#combine the seperate dataframes in the list into one dataframe
statistics_model = pd.concat(df_list, axis=0)
#combine all dataframes from the dictionary into one dataframe
results_model = pd.concat(model_results, axis=1)

# Create an Excel with the results and the statistics of the models
with pd.ExcelWriter(params.statisticsDir + "Model_results.xlsx") as writer:
    # Write each dataframe to a different worksheet.
    results_model.to_excel(writer, sheet_name="Model_results")
    statistics_model.to_excel(writer, sheet_name="Model_statistics")

'''
Apply the model to the Loss of Industry Scenario
'''

# Load the LoI scenario data, select the subset of columns for each phase and rename them
#to fit the names used in the model formular
LoI_data, LoI_phase1, LoI_phase2 = {}, {}, {}
for crop in crops:
    LoI_data[crop] = pd.read_csv(
        params.LoIDataDir + crop + "_LoI_data.gzip", index_col=0, compression="gzip"
    )
    # select the rows from LoI_data which contain the independent variables for phase 1
    LoI_phase1[crop] = LoI_data[crop].iloc[:, [3, 4, 5, 6, 9, 12, 13, 14]]
    # rename the columns according to the names used in the model formular
    LoI_phase1[crop] = LoI_phase1[crop].rename(
        columns={
            "p_fertilizer_y1": "p_fertilizer",
            "n_total_y1": "n_total",
            "pesticides_y1": "pesticides",
            "irrigation_LoI": "irrigation_tot",
            },
        errors="raise",
        )
    # select the rows from LoI_data which contain the independent variables for phase 2
    LoI_phase2[crop] = LoI_data[crop].iloc[:, [4, 5, 6, 8, 9, 10, 15, 16]]
    # rename the columns according to the names used in the model formular
    LoI_phase2[crop] = LoI_phase2[crop].rename(
        columns={
            "p_fertilizer_y2": "p_fertilizer",
            "manure_LoI": "n_total",
            "pesticides_y2": "pesticides",
            "mechanized_y2": "mechanized",
            "irrigation_LoI": "irrigation_tot",
            },
        errors="raise",
        )      

#function to get the prediction as well as the standard error, the upper and lower
#values for a specified model, confidence interval and a given dataset
#replaces standard error with a half confidence interval
#fit: a fitted model instance
#data: a dataset containing the necessary factors for the specified model
#alpha: the confidence level for the confidence interval
def prediction_to_df (fit, data, alpha):
    df = fit.get_prediction(data).summary_frame(alpha=alpha)
    #set index to the index of the originial data
    df.index = data.index
    df.iloc[:,1] = (abs(df.iloc[:,3] - df.iloc[:,2]))/2
    df = df.rename(columns={'mean_se': "mean_1/2_ci"}, errors="raise")
    return df

#function to correct a model inaccuracy: the model predicts higher yield values
#for some points in the LoI scenarios which is likely due to its inability to 
#properly model lower yields
#as a correction measure, the change in relation to the original yields is
#calculated and all predictions which result in positive relative change
#are set to the original yield data resulting in zero relative change
def reset_postive_change(predictions, data):
    change = (predictions['mean']-data)/data
    #select all predictions which show a relative change below or equal to 0
    predict_negative = predictions.loc[change <=0]
    #select the values of the original yield data where the predictions
    #result in a relative change above 0
    substitutes = data.loc[change>0]
    sub_dict = {'mean': substitutes,
            #fill the remaining three columns with NoData values as there is
            #no uncertainty information on the substituted values
           'mean_1/2_ci':[-9999] * len(substitutes),
           'mean_ci_lower':[-9999] * len(substitutes),
           'mean_ci_upper':[-9999] * len(substitutes)}
    sub_df = pd.DataFrame(sub_dict)
    result = pd.concat([predict_negative, sub_df], verify_integrity=True).sort_index()
    return result

phases = ['phase_1', 'phase_2']
columns = ['mean', 'mean_1/2_ci','mean_ci_lower', 'mean_ci_upper']
phase_data = {'phase_1':LoI_phase1,
          'phase_2':LoI_phase2}

#apply the formulas defined above for each crop, calculate relative change and its uncertainty
#measures for each crop
LoI_predictions_raw, LoI_predictions, relative_change = {}, {}, {}
for crop in crops:
    LoI_predictions_raw[crop], LoI_predictions[crop], relative_change[crop] = {}, {}, {}
    for phase in phases:
        #get predictions for each crop and each phase and save them in dataframes
        LoI_predictions_raw[crop][phase] = prediction_to_df(fit[crop], phase_data[phase][crop], 0.05)
        #correct the predictions for positive relative changes
        LoI_predictions[crop][phase] = reset_postive_change(LoI_predictions_raw[crop][phase], model_data[crop]['Yield'])
        #create a dataframe to calculate relative change and the corresponding uncertainty measures
        relative_change[crop][phase] = pd.DataFrame(columns=columns, index=LoI_predictions[crop][phase].index)
        #calculate the relative change and the correpsonding uncertainty measures for each phase and crop
        for column in columns:
            relative_change[crop][phase][column] = np.where(
                LoI_predictions[crop][phase][column] == -9999, -9999,
                (LoI_predictions[crop][phase][column] - model_data[crop]['Yield']) / model_data[crop]['Yield'])  
        relative_change[crop][phase]['mean_1/2_ci'] = np.where(
            LoI_predictions[crop][phase]['mean_1/2_ci'] == -9999, -9999,
            (abs(relative_change[crop][phase]['mean_ci_upper'] - relative_change[crop][phase]['mean_ci_lower']))/2
            )

'''
Create ascii files for the mean, upper and lower bound of the yield and the relative change
#for each crop and phase 
'''

#function to combine a column of a dataset with the coordinates for the whole world,
#and save it as an ascii file
def createASCII(df, column, coordinates, file):
    #combine the world coordinates with the column of a dataframe
    df_asc = pd.concat([coordinates, df[column]], axis='columns')
    #create the data directory if it doesn't exist already
    os.makedirs(params.outputAsciiDir, exist_ok=True)
    #create an ascii file from the concated dataframe
    utilities.create5minASCIIneg(
    df_asc, column, params.outputAsciiDir + '/' + file
)

column_names = ['mean', 'mean_ci_lower', 'mean_ci_upper']
column_codes = {'mean': 'mean',
                'mean_ci_lower': 'ci_lower',
                'mean_ci_upper': 'ci_upper'}
types = ['LoI_predictions', 'relative_change']
pred_data = {'LoI_predictions': [LoI_predictions, 'yield'],
         'relative_change': [relative_change, 'rc']}

#apply the above specified function for each crop and phase
for crop in crops:
    for phase in phases:
        for column in column_names:
            for typ in types: 
                #create the file name out of the crop, the phase, yield or relative change and if it's
                #the mean, upper or lower bound
                file_name = crop + '_' + phase + '_' + pred_data[typ][1] + '_' + column_codes[column]
                createASCII(pred_data[typ][0][crop][phase], column, coordinates, file_name)
            
"""
Calculate descriptive statistics for each crop for SPAM2010, fitted values, Phase 1 and Phase 2
"""

# specify columns
steps = ["Yield_SPAM2010", "Yield_fitted_values", "Yield_phase1", "Yield_phase2",
         'RelativeChange_phase1', 'Relative_Change_phase2']

#complile all datasets which are needed for the statistics calculations
#in a new dataframe
weights_change, fitted, stat_data = {}, {}, {}
for crop in crops:
    #calculate the weights for calculating the weighted mean of the relative change
    weights_change[crop] = model_data[crop]["Yield"] * model_data[crop]["area"]
    #calculate the fitted values for each crop
    fitted[crop] = pd.concat([fit[crop].fittedvalues, val[crop]]).sort_index()
    stat_data[crop] = pd.concat(
    [
        model_data[crop]["Yield"],
        fitted[crop],
        LoI_predictions[crop]['phase_1']["mean"],
        LoI_predictions[crop]['phase_2']["mean"],
        relative_change[crop]['phase_1']['mean'],
        relative_change[crop]['phase_2']['mean'],
    ],
    axis=1,
)
    stat_data[crop].columns = steps

#function to calculate a weighted standard error for given data: standard computation
#method except that it takes the weighted sample mean as basis
#data: series of values
#weights: series of values containing the weights for the calculation,
#same length as data
def weighted_sem(data, weights):
    sample_mean = stat_ut.weighted_average(data, weights)
    N = len(data)
    sd = np.sqrt(sum((data - sample_mean)**2)/(N-1))
    sem = sd/np.sqrt(N)
    return sem

#function to calculate half of a weighted confidence interval for given data:
#standard computation method except that it takes the weighted standard error as basis
#data: series of values
#weights: series of values containing the weights for the calculation,
#same length as data
#level: desired confidence level, default: 0.05
def weighted_half_ci(data, weights, level=0.05):
    df = len(data) - 1
    t = stats.t.ppf(level, df)
    ci = round(abs(t * weighted_sem(data, weights)), 4)
    return ci

#calculate descriptive statistics (Weighted mean, weighted confidence interval, minimum,
#maximum, Total yearly production) for each crop and each step
prediction_stats = {}
for crop in crops:
    prediction_stats[crop] = pd.DataFrame(stat_data[crop].apply(
                lambda x: stat_ut.weighted_average(
                    x, weights=model_data[crop]["area"], dropna=True
                )), columns=['Weighted_Mean'])
    prediction_stats[crop]['Weighted_Mean'].iloc[4:6]= stat_data[crop].iloc[:,4:6].apply(
                lambda x: stat_ut.weighted_average(
                    x, weights=weights_change[crop]))
    prediction_stats[crop]['Mean_1/2_ci'] = stat_data[crop].apply(
                lambda x: weighted_half_ci(
                    x, weights=model_data[crop]["area"]))
    prediction_stats[crop]['Mean_1/2_ci'].iloc[4:6] = stat_data[crop].iloc[:,4:6].apply(
                lambda x: weighted_half_ci(
                    x, weights=weights_change[crop]))
    prediction_stats[crop]['Minimum'] = round(stat_data[crop].min(), 2)
    prediction_stats[crop]['Maximum'] = round(stat_data[crop].max(), 2)
    prediction_stats[crop]['Production'] = stat_data[crop].sum()
    prediction_stats[crop]['Production'].iloc[4:6] = np.nan
prediction_statistics = pd.concat(prediction_stats).sort_index(level=0, sort_remaining=False)
 
'''
Calculate continent level statistics for each crop and each step
'''

#calculate the mean yield and relative change respectively for each continent,
#crop and step

continent_names = {1:'Africa', 
                   2:'Asia', 
                   3:'Oceania',
                   4:'North America',
                   5:'Europe',
                   6:'South America'}
   
continent_stats = {}
for crop in crops:
    #rename continent levels
    LoI_data[crop]['continents'].replace(continent_names,inplace=True)
    continent_stats[crop] = stat_ut.weighted_mean_zonal(
        stat_data[crop], LoI_data[crop]["continents"], model_data[crop]["area"]
        )
    continent_stats[crop].iloc[:,4:6] = stat_ut.weighted_mean_zonal(
        stat_data[crop].iloc[:,4:6], LoI_data[crop]["continents"], weights_change[crop]
        )

#unsure if I should add the ci (could multiindex index like so: 1 -> mean, 1/2 ci, 2 -> mean, 1/2 ci, etc.)
continent_statistics = pd.concat(continent_stats).sort_index(level=0, sort_remaining=False)

# Create an Excel with the results and the statistics of the models
with pd.ExcelWriter(params.statisticsDir + "Prediction_statistics.xlsx") as writer:
    # Write each dataframe to a different worksheet.
    prediction_statistics.to_excel(writer, sheet_name="Prediction_statistics")
    continent_statistics.to_excel(writer, sheet_name="Continent_statistics")