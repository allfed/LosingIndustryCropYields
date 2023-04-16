# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:25:39 2022

@author: Jess
"""
import os
import sys
import src.utilities.params as params  # get file location and varname parameters for
import src.utilities.utilities as utilities  # get file location and varname parameters for
import src.utilities.stat_ut as stat_ut  # get file location and varname parameters for
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
#######################################################################
########### Regression Calibration, Validation and Residuals###########
#######################################################################

crops = {"Corn", "Rice", "Soybean", "Wheat"}

"""
Load the clean dataset for each crop and split the data into a validation and a calibration dataset
"""

# take the cleaned dataset as a basis to calculate the conditions for the LoI scenario in phase 1 and 2
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
    
coordinates = pd.read_pickle(params.LoIDataDir  + "Coordinates.pkl", compression="zip")

"""
Calibrate the Regression model, calculate fit statistics and validate the model
against the validation dataset
"""

model, fit, val = {}, {}, {}
for crop in crops:
    # determine model with a gamma distribution
    model[crop] = smf.glm(
        formula="Yield ~ n_total + p_fertilizer + irrigation_tot + mechanized + pesticides +  C(thz_class) + \
                  C(mst_class) + C(soil_class)",
              data=fit_data[crop],
              family=sm.families.Gamma(link=sm.families.links.log),
              )
    # Fit models
    fit[crop] = model[crop].fit()
    # let the model predict yield values for the validation data
    val[crop] = fit[crop].predict(val_factors[crop])

results, statis, df_list = {}, {}, []
for crop in crops:
    #collect coefficients, 95% Confidence Intervals, odds ratios and p-values in a dataframe
    results[crop] = pd.DataFrame(fit[crop].params, columns=["Coefficients"])
    results[crop]['+-95%Confidence_Interval'] = fit[crop].bse * 2
    results[crop]['Odds_ratios'] = np.exp(fit[crop].params)
    results[crop]['p-value'] = fit[crop].pvalues
    #calculate McFadden's roh² and the Root Mean Gamma Deviance (RMGD) for 
    #fitted values and predicted values based on the validation dataset
    #calculate the AIC and BIC for each model
    statis[crop] = [], []
    statis[crop][0].append(d2_tweedie_score(fit_data[crop]["Yield"], fit[crop].fittedvalues, power=2))
    statis[crop][0].append(np.sqrt(mean_tweedie_deviance(fit_data[crop]["Yield"], fit[crop].fittedvalues, power=2)))
    statis[crop][0].append(fit[crop].aic)
    statis[crop][0].append(fit[crop].bic_llf)
    statis[crop][1].append(d2_tweedie_score(val_data[crop]["Yield"], val[crop], power=2))
    statis[crop][1].append(np.sqrt(mean_tweedie_deviance(val_data[crop]["Yield"], val[crop], power=2)))
    statis[crop][1].extend([np.nan]*2)
    #combine the lists into a dataframe with a multiindex and create a list of dataframes containing one df for each crop
    df = pd.DataFrame(statis[crop], index=['Calibration', 'Validation'], columns=['McFaddens_roh', 'RootMeanGammaDeviance', 'AIC', 'BIC'])
    df.index = pd.MultiIndex.from_product([[crop], df.index])
    df_list.append(df)
#combine the seperate datafremes into one dataframe
statistics_model = pd.concat(df_list, axis=0)
results_model = pd.concat(results, axis=1)

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

### Phase 1 ###

def prediction_to_df (fit, data, alpha):
    df = fit.get_prediction(data).summary_frame(alpha=alpha)
    df.index = data.index
    #the next two lines could be taken out , if I decide not to calculate the +- for each point because
    #I don't know the exact multiplication value anyways
    df.iloc[:,1] = (abs(df.iloc[:,3] - df.iloc[:,2]))/2
    df = df.rename(columns={'mean_se': "mean_1/2_ci"}, errors="raise")
    return df

def reset_postive_change(predictions, data):
    change = (predictions['mean']-data)/data
    predict_negative = predictions.loc[change <=0]
    substitutes = data.loc[change>0]
    sub_dict = {'mean': substitutes,
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

LoI_predictions_raw, LoI_predictions, relative_change = {}, {}, {}
for crop in crops:
    LoI_predictions_raw[crop], LoI_predictions[crop], relative_change[crop] = {}, {}, {}
    for phase in phases:
        LoI_predictions_raw[crop][phase] = prediction_to_df(fit[crop], phase_data[phase][crop], 0.05)
        LoI_predictions[crop][phase] = reset_postive_change(LoI_predictions_raw[crop][phase], model_data[crop]['Yield'])
        relative_change[crop][phase] = pd.DataFrame(columns=columns, index=LoI_predictions[crop][phase].index)
        for column in columns:
            relative_change[crop][phase][column] = np.where(
                LoI_predictions[crop][phase][column] == -9999, -9999,
                (LoI_predictions[crop][phase][column] - model_data[crop]['Yield']) / model_data[crop]['Yield'])  
        relative_change[crop][phase]['mean_1/2_ci'] = np.where(
            LoI_predictions[crop][phase]['mean_1/2_ci'] == -9999, -9999,
            (abs(relative_change[crop][phase]['mean_ci_upper'] - relative_change[crop][phase]['mean_ci_lower']))/2
            )

def createASCII(df, column, coordinates, file):
    df_asc = pd.concat([coordinates, df[column]], axis='columns')
    os.makedirs(params.outputAsciiDir, exist_ok=True)
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

for crop in crops:
    for phase in phases:
        for column in column_names:
            for typ in types: 
                file_name = crop + '_' + phase + '_' + pred_data[typ][1] + '_' + column_codes[column]
                createASCII(pred_data[typ][0][crop][phase], column, coordinates, file_name)
            
"""
Descriptive Statistics for each crop for SPAM2010, fitted values, Phase 1 and Phase 2
"""

# specify columns and apply function for all crops
metrics = ["Total_Area(ha)", "Number_Rows"]
#steps = {"raw": 0, "step1": 1, "step2": 2, "clean": 3, "outliers": 4}
steps = ["SPAM2010", "fitted_values", "phase_1", "phase_2"]
cat = {
    "mechanized": 1,
    "thz_class": 2,
    "mst_class": 3,
    "soil_class": 4,
    "continents": 5,
}
columns_stat = {
    "Yield": 0,
    "n_fertilizer": 1,
    "p_fertilizer": 2,
    "n_manure": 3,
    "n_total": 4,
    "pesticides": 5,
    "irrigation_tot": 6,
    "irrigation_rel": 7,
    "mechanized": 8,
    "thz_class": 9,
    "mst_class": 10,
    "soil_class": 11,
    "continents": 12,
}
data = {
    "SPAM2010": [data_raw, columns_stat],
    "fitted_values": [data_step1, columns_stat],
    "phase_1": [data_step2, columns_stat],
    "phase_2": [data_clean, columns_stat],
}

continent_stats, continent_stats_yield, weights_change, fitted, instances_yield, instances_change = {}, {}, {}, {}, {}, {}
for crop in crops:
    weights_change[crop] = model_data[crop]["Yield"] * model_data[crop]["area"]
    fitted[crop] = pd.concat([fit[crop].fittedvalues, val[crop]]).sort_index()
    instances_yield[crop] = pd.concat(
    [
        model_data[crop]["Yield"],
        fitted[crop],
        LoI_predictions[crop]['phase_1']["mean"],
        LoI_predictions[crop]['phase_2']["mean"],
    ],
    axis=1,
)
    instances_yield[crop].columns = steps
    instances_change[crop] = pd.concat(
    [
        relative_change[crop]['phase_1'],
        relative_change[crop]['phase_2'],
    ],
    axis=1,
)
    instances_change[crop].columns = phases
    continent_stats_yield[crop] = stat_ut.weighted_mean_zonal(instances_yield[crop], LoI_data[crop]["continents"], model_data[crop]["area"])
    
fit[crop].summary()
6.4006 + 0.022*1.96
6.4006 + 0.022*2
stats.sem(model_data[crop]['Yield'])
stat_ut.weighted_sem(model_data[crop]['Yield'], model_data[crop]['area'])
def weighted_sem(data, weights):
    sample_mean = stat_ut.weighted_average(model_data[crop]['Yield'], model_data[crop]['area'])
    N = len(model_data[crop]['Yield'])
    sd = np.sqrt(sum((model_data[crop]['Yield'] - sample_mean)**2)/(N-1))
    sem = sd/np.sqrt(N)
    return sem


LoI_data[crop]["continents"].unique()
result = pd.concat(df_list, axis=1).transpose()

descriptive_stats, desc_stats = {}, {}
for crop in crops:
    descriptive_stats[crop] = {}
    df_list = []
    for step in steps:
        descriptive_stats[crop][step] = pd.DataFrame(
            data[step][0][crop].loc[:, data[step][1]].max(), columns=["2_Maximum"]
        )
        descriptive_stats[crop][step]["1_Minimum"] = (
            data[step][0][crop].loc[:, data[step][1]].min()
        )
        descriptive_stats[crop][step]["0_Weighted_Mean_Mode"] = (
            data[step][0][crop]
            .loc[:, data[step][1]]
            .apply(
                lambda x: stat_ut.weighted_average(
                    x, weights=data[step][0][crop]["area"], dropna=True
                )
            )
        )
        if step == "outliers":
            descriptive_stats[crop][step]["3_Outlier_threshold"] = out_threshold[
                crop
            ].values()
            descriptive_stats[crop][step].loc[data[step][1], "4_Number_Outliers"] = (
                data[step][0][crop].loc[:, data[step][1]].notna().sum()
            )
        else:
            descriptive_stats[crop][step]["5_NaN_count"] = (
                data[step][0][crop].loc[:, data[step][1]].isna().sum()
            )
            descriptive_stats[crop][step]["6_0_count"] = (
                data[step][0][crop].loc[:, data[step][1]] == 0
            ).sum()
            descriptive_stats[crop][step].loc[cat, "0_Weighted_Mean_Mode"] = (
                data[step][0][crop]
                .loc[:, cat]
                .apply(
                    lambda x: stat_ut.weighted_mode(
                        x, weights=data[step][0][crop]["area"], dropna=True
                    )
                )
            )
        descriptive_stats[crop][step].columns = pd.MultiIndex.from_product(
            [descriptive_stats[crop][step].columns, [step]]
        )
        df_list.append(descriptive_stats[crop][step])
    desc_stats[crop] = pd.concat(df_list, axis=1).sort_index(
        level=0, axis=1, sort_remaining=False
    )

# calculate statistics for yield and change rate

# calculate weights for mean change rate calculation dependent on current yield
# and current crop area in a cell
ww = LoI_welim["Y"] * LoI_welim["area"]

# calculate weighted mean, min and max of predicted yield (1) including postive change rates
wmean_y1_weigh = round(np.average(w_yield_y1, weights=LoI_welim["area"]), 2)
wmax_y1 = w_yield_y1.max()
wmin_y1 = w_yield_y1.min()
# (2) excluding postive change rates
wmean_y1_0 = round(np.average(w_y1_y0, weights=LoI_welim["area"]), 2)
wmax_y10 = w_y1_y0.max()
wmin_y10 = w_y1_y0.min()

# change rate
wmean_y1c_weigh = round(np.average(w_y1_change, weights=ww), 2)
wmax_y1c = w_y1_change.max()
wmin_y1c = w_y1_change.min()

#Phase 2
# calculate statistics for yield and change rate

# calculate weighted mean, min and max of predicted yield (1) including postive change rates
wmean_y2_weigh = round(
    np.average(w_yield_y2, weights=LoI_welim["area"]), 2
)  # 1799.3kg/ha
wmax_y2 = w_yield_y2.max()  # 4310.70g/ha
wmin_y2 = w_yield_y2.min()  # 579.24kg/ha
# (2) excluding postive change rates
wmean_y2_0 = round(np.average(w_y2_y0, weights=LoI_welim["area"]), 2)  # 1640.8 kg/ha
wmax_y20 = w_y2_y0.max()  # 4310.70kg/ha
wmin_y20 = w_y2_y0.min()  # 74.1kg/ha

# calculate weighted mean, min and max of predicted change rate (1) including postive change rates
wmean_y2c_weigh = round(np.average(w_y2_change, weights=ww), 2)  # -0.43 (~-43%)
wmax_y2c = w_y2_change.max()  # 34.56 (~+3456%)
wmin_y2c = w_y2_change.min()  # -0.9394 (~-94%)
# (2) excluding postive change rates
wmean_y2c0_weigh = round(np.average(w_c0["w_y2_c0"], weights=ww), 2)  # -0.46
wmean_y1c0_weigh = round(np.average(w_c0["w_y1_c0"], weights=ww), 2)  # -0.35

"""
Statistics to compare current SPAM2010 yield with (1) current fitted values,
(2) phase 1 and (3) phase 2 predictions
"""

## calculate statistics for current yield ##

# SPAM2010 yield: weighted mean, max, min, total yearly production
dw0_mean = round(np.average(dw0_elim["Y"], weights=dw0_elim["area"]), 2)
dw0_max = dw0_elim["Y"].max()
dw0_min = dw0_elim["Y"].min()
dw0_prod = (dw0_elim["Y"] * dw0_elim["area"]).sum()
# fitted values for current yield based on Gamma GLM: weighted mean, max and min
test = w_fit_elimg.fittedvalues
w_fit_mean = round(
    np.average(w_fit_elimg.fittedvalues, weights=dwheat_fit_elim["area"]), 2
)
w_fit_max = w_fit_elimg.fittedvalues.max()
w_fit_min = w_fit_elimg.fittedvalues.min()
w_fit_prod = (w_fit_elimg.fittedvalues * dwheat_fit_elim["area"]).sum()

## calculate statistics for both phases ##

# phase 1: calculate the percentage of current yield/production which will be achieved
# in phase 1 as predicted by the GLM, calculate total production in phase 1
# (1) including positive change rates and
w_y1_pery = (
    wmean_y1_weigh / dw0_mean
)  # sanity check, should be 100-mean change rate: ~72.2% of current average yield
w_y1_prod = (w_yield_y1 * LoI_welim["area"]).sum()
w_y1_perp = w_y1_prod / dw0_prod
# (2) with positive change rates set to 0
w_y10_pery = (
    wmean_y1_0 / dw0_mean
)  # sanity check, should be 100-mean change rate: ~65.5% of current average yield
w_y10_prod = (w_y1_y0 * LoI_welim["area"]).sum()
w_y10_perp = w_y10_prod / dw0_prod

# phase 2: calculate the percentage of current yield/production which will be achieved
# in phase 2 as predicted by the GLM, calculate total production in phase 1
# (1) including positive change rates and
w_y2_pery = (
    wmean_y2_weigh / dw0_mean
)  # sanity check, should be 100-mean change rate: ~ 57.4% of current average yield
w_y2_prod = (w_yield_y2 * LoI_welim["area"]).sum()
w_y2_perp = w_y2_prod / dw0_prod
# (2) with positive change rates set to 0
w_y20_pery = (
    wmean_y2_0 / dw0_mean
)  # sanity check, should be 100-mean change rate: ~53.05% of current average yield
w_y20_prod = (w_y2_y0 * LoI_welim["area"]).sum()
w_y20_perp = w_y20_prod / dw0_prod

# print the relevant statistics of SPAM2010, fitted values, phase 1 and phase 2
# predictions in order to compare them
# 1st column: weighted mean
# 2nd row: total crop production in one year
# 3rd row: maximum values
# 4th row: minimum values
# last two rows comprise statistics for phase 1 and 2 (1) including positive change rates
# and (2) having them set to 0
# 5th row: percentage of current production achieved in each phase
# 6th row: mean yield change rate for each phase
print(
    dw0_mean,
    w_fit_mean,
    wmean_y1_weigh,
    wmean_y2_weigh,
    dw0_prod,
    w_fit_prod,
    w_y1_prod,
    w_y2_prod,
    dw0_max,
    w_fit_max,
    wmax_y1,
    wmax_y2,
    dw0_min,
    w_fit_min,
    wmin_y1,
    wmin_y2,
    w_y1_perp,
    w_y2_perp,
    w_y10_perp,
    w_y20_perp,
    wmean_y1c_weigh,
    wmean_y2c_weigh,
    wmean_y1c0_weigh,
    wmean_y2c0_weigh,
)

"""
Weighted Zonal Statistics for the continents
"""
'''
continents = pd.read_csv(params.geopandasDataDir + "Continents.csv")
cont = continents.iloc[w_yield_y1.index]
t = pd.concat(
    [cont, LoI_welim], axis=1, join="inner"
)  # sanity check to see if coordinates match
'''
zon_stat = pd.concat(
    [
        cont["lats"],
        cont["lons"],
        cont["continent"],
        dw0_elim["Y"],
        LoI_welim["area"],
        w_yield_y1,
        w_y1_y0,
        w_y1_change,
        w_c0["w_y1_c0"],
        w_yield_y2,
        w_y2_y0,
        w_y2_change,
        w_c0["w_y2_c0"],
    ],
    axis=1,
    join="inner",
)
zon_stat = zon_stat.rename(
    columns={
        0: "w_yield_y1",
        1: "w_y1_y0",
        2: "w_y1_change",
        3: "w_yield_y2",
        4: "w_y2_y0",
        5: "w_y2_change",
    },
    errors="raise",
)

cont_lev = np.sort(zon_stat["continent"].unique())

cont_0 = zon_stat.loc[zon_stat["continent"] == 0]
new_val = pd.Series(
    [4, 6, 5, 5, 5, 5, 5, 1, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2], index=cont_0.index
)
# new_val.name = 'new_cont'
new_cont = zon_stat["continent"]
new_cont = new_cont.drop(new_val.index)
new_cont = new_cont.append(new_val).sort_index()

zon_c = pd.concat(
    [
        zon_stat["w_y1_change"],
        zon_stat["w_y2_change"],
        zon_stat["w_y1_c0"],
        zon_stat["w_y2_c0"],
    ],
    axis=1,
)

wm_t = stat_ut.weighted_mean_zonal(zon_c, new_cont, ww)


"""
save the predicted yields and the yield change rates for each phase
"""

# combine yields and change rates of both phases with the latitude and longitude values
LoI_wheat = pd.concat(
    [
        wheat_yield["lats"],
        wheat_yield["lons"],
        w_yield_y1,
        w_y1_change,
        w_yield_y2,
        w_y2_change,
        w_c0,
    ],
    axis="columns",
)
LoI_wheat = LoI_wheat.rename(
    columns={0: "w_yield_y1", 1: "w_y1_change", 2: "w_yield_y2", 3: "w_y2_change"},
    errors="raise",
)
# save the dataframe in a csv
LoI_wheat.to_csv(params.geopandasDataDir + "LoIWheatYieldHighRes.csv")

# save the yield for phase 1 and 2 and the change rate for phase 1 and 2 with and without positive rates
# as ASCII files
utilities.create5minASCIIneg(
    LoI_wheat, "w_y1_change", params.asciiDir + "LoIWheatYieldChange_y1"
)
utilities.create5minASCIIneg(
    LoI_wheat, "w_yield_y1", params.asciiDir + "LoIWheatYield_y1"
)
utilities.create5minASCIIneg(
    LoI_wheat, "w_y2_change", params.asciiDir + "LoIWheatYieldChange_y2"
)
utilities.create5minASCIIneg(
    LoI_wheat, "w_yield_y2", params.asciiDir + "LoIWheatYield_y2"
)
utilities.create5minASCIIneg(
    LoI_wheat, "w_y1_c0", params.asciiDir + "LoIWheatYieldChange_0y1"
)
utilities.create5minASCIIneg(
    LoI_wheat, "w_y2_c0", params.asciiDir + "LoIWheatYieldChange_0y2"
)
