"""

File containing the code to prepare the input data and perform a multiple regression
on yield for wheat at 5 arcmin resolution


Jessica Mörsdorf
jessica@allfed.info
jessica.m.moersdorf@umwelt.uni-giessen.de

"""

import os
import sys

import src.utilities.params as params  # get file location and varname parameters for
import src.utilities.stat_ut as stat_ut  # get file location and varname parameters for

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

params.importAll()

crops = ["Corn", "Rice", "Soybean", "Wheat"]


"""
Load raw data and eliminate all rows with an area below 100 ha
"""
data_raw, data_step1 = {}, {}
for crop in crops:
    data_name = "{}_raw".format(crop)
    data_raw[crop] = pd.read_pickle(
        params.cropDataDir + data_name + ".pkl", compression="zip"
    )
    data_raw[crop]["irrigation_rel"] = data_raw[crop]["irrigation_rel"].fillna(0)
    data_raw[crop] = data_raw[crop].replace(
        [data_raw[crop]["n_fertilizer"].min(), -9], np.nan
    )
    data_raw[crop] = data_raw[crop].replace(
        {"continents": 0, "irrigation_tot": -0},
        {"continents": np.nan, "irrigation_tot": 0},
    )
    data_step1[crop] = data_raw[crop].loc[data_raw[crop]["area"] > 100]

print("Done reading crop data and eliminating all rows below 100 ha")

"""
Calculate area of raw datasets for specified columns to use them in LoI_scenario_data.py
"""


def calculate_area(data, columns):
    dict_area = {}
    for col in columns:
        dict_area[col] = data["area"].loc[data[col].notna()].sum().astype("int")
    return dict_area


# specify columns and apply function for all crops
columns = ["n_fertilizer", "p_fertilizer", "pesticides"]
area_stat = {crop: calculate_area(data_raw[crop], columns) for crop in crops}
# convert dict to dataframe and save to csv
area_col = pd.DataFrame.from_dict(area_stat)
os.makedirs(params.statisticsDir, exist_ok=True)
area_col.to_csv(params.statisticsDir + "Raw_Column_Area.csv")

print("Done calculating total crop area and saving it to csv")

"""
Combine some of the levels of AEZ classes, fill or eliminate missing data
in soil, n+p fertilizer, n_total, pesticides, mechanized and irrigation_rel
"""


# function to fill classes with unreasonable values (for cropland) in soil class
# and combine some of the levels of the thz & mst class to ensure that levels with
# few datapoints don't introduce imbalance into the data
def clean_aez(data):
    data_aez = data.copy()
    # replace 0s, 7s & 8s in the soil class with NaN values so they can be handled with the .fillna method
    data_aez["soil_class"] = data_aez["soil_class"].replace([0, 7, 8], np.nan)
    # fill in the NaN vlaues in the dataset with a forward filling method
    # (replacing NaN with the value in the cell before)
    data_aez = data_aez.fillna(downcast="infer", method="ffill")
    # replace 8,9 & 10 with 7 in the thz class to combine all 3 classes into 1 Temp,cool-Arctic class
    # repalce 2 with 1 and 7 with 6 in the mst class to compile them into 1 joined class each
    data_aez["thz_class"] = data_aez["thz_class"].replace([8, 9, 10], 7)
    data_aez["mst_class"] = data_aez["mst_class"].replace({2: 1, 7: 6})
    return data_aez


# function to fill missing values in the n_fertilizer, p_fertilizer and n_total columns
def clean_fertilizer(data):
    data_fertilizer = data.copy()
    no_data_value = data_fertilizer["n_fertilizer"].min()
    # replace remaining no data values in fertilizer datasets with NaN, then fill them
    data_fertilizer = data_fertilizer.replace(
        {"n_fertilizer": no_data_value, "p_fertilizer": no_data_value}, np.nan
    )
    data_fertilizer = data_fertilizer.fillna(method="ffill")
    # replace NaN values in n_total with sum of the newly filled n_fertilizer and n_manure values
    data_fertilizer.loc[data_fertilizer["n_total"] < 0, "n_total"] = (
        data_fertilizer["n_fertilizer"] + data_fertilizer["n_manure"]
    )
    return data_fertilizer


# for each crop: remove rows with missing data in pesticides and mechanized columns, replace missing values in
# irrigation_rel with 0 and apply the above defined functions
# replace NaN values with -9 so that fillna can be used on the entire dataframe in the next step
data_step2 = {crop: data_step1[crop].replace(np.nan, -9) for crop in crops}
for crop in crops:
    # fill NaN values in soil_class and combine categories in thz & mst class
    data_step2[crop] = clean_aez(data_step2[crop])
    # fill NaN values in n & p fertilizer and n total columns
    data_step2[crop] = clean_fertilizer(data_step2[crop])
    # Eliminate the rows without data in the pesticides and mechanized columns
    data_step2[crop] = data_step2[crop].query("pesticides > -9 and mechanized > -9")
    # replace -9 in continent column with np.nan
    data_step2[crop] = data_step2[crop].replace(-9, np.nan)

print("Done combining AEZ classes, filling or eliminating missing data")

"""
Calculate and eliminate outliers (values above the 99th/99.9th quantile) from the dataset
and extract the outliers to calculate outlier statistics in a later step
"""


# function to remove all rows from a dataframe where the values of the specified
# columns surpass the specified threshold
def eliminate_outliers(data, factors, thresholds):
    data_outliers = data.loc[
        np.logical_and.reduce([data[factor] < thresholds[factor] for factor in factors])
    ]
    return data_outliers


# function to extract all rows from a dataframe where the values of the specified
# columns surpass the specified threshold and combine the results with the area
# column in one dataframe (area is needed for weighted mean calculations in a later step)
def extract_outliers(data, factors, thresholds):
    outlier = {}
    for factor in factors:
        outlier[factor] = data[factor].loc[data[factor] >= thresholds[factor]]
    outliers = pd.DataFrame(outlier, columns=factors)
    results = pd.concat(
        [outliers, data["area"][data["area"].index.isin(outliers.index)]],
        axis="columns",
    )
    return results


# select the columns of the dataframe where outliers will be calculated
factors = {
    "Yield": 5,
    "n_fertilizer": 0,
    "p_fertilizer": 3,
    "n_manure": 1,
    "n_total": 2,
    "pesticides": 4,
}

# for each crop: calculate the outlier thresholds and apply the above functions
out_threshold, data_step3, outliers = {}, {}, {}
for crop in crops:
    # combine variables where to calculate the outliers and the 99.9th quantile of each variable into a dictionary
    out_threshold[crop] = dict(zip(factors, data_step2[crop][factors].quantile(0.999)))
    # replace the value for n_manure with the 99th quantile
    out_threshold[crop]["n_manure"] = data_step2[crop]["n_manure"].quantile(0.99)
    data_step3[crop] = eliminate_outliers(
        data_step2[crop], factors, out_threshold[crop]
    )
    outliers[crop] = extract_outliers(data_step2[crop], factors, out_threshold[crop])

print("Done extracting and eliminating outliers")

"""
Replace No Data Values in continent column with corresponding continent value
"""

# Create lists with the continent values for the missing data points
fill_values_Corn = (
    [4] * 11 + [6, 6, 4, 4, 4, 6, 1, 5, 1, 1, 1, 5, 1, 1, 1, 1, 1] + [2] * 39
)
fill_values_Rice = [6] * 12 + [1] * 10 + [2] * 132
fill_values_Soybean = [4] * 3 + [2] * 8
fill_values_Wheat = [4, 6, 5, 5, 5, 5, 5, 1, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2]

# create a dictionary with the NoData index for each crop
NoData_Index = {
    crop: data_step3[crop].loc[data_step3[crop]["continents"].isna()].index
    for crop in crops
}

# store the lists in a dictionary
continents_NoData = {
    "Corn": pd.Series(fill_values_Corn, index=NoData_Index["Corn"]),
    "Rice": pd.Series(fill_values_Rice, index=NoData_Index["Rice"]),
    "Soybean": pd.Series(fill_values_Soybean, index=NoData_Index["Soybean"]),
    "Wheat": pd.Series(fill_values_Wheat, index=NoData_Index["Wheat"]),
}

data_clean = {crop: data_step3[crop].copy() for crop in crops}
for crop in crops:
    # set the cells where continent==0 to the values stored in the continents_NoData dict
    data_clean[crop]["continents"] = data_clean[crop]["continents"].fillna(
        continents_NoData[crop], downcast="infer"
    )
    # save the clean dataset to csv
    os.makedirs(params.modelDataDir, exist_ok=True)

    data_clean[crop].to_csv(
        params.modelDataDir + crop + "_data.gzip", compression="gzip"
    )

print(
    "Done replacing no data values in the continent column and saving the clean dataset to file"
)

############################################################################
###I think the dummy-code can most likely go if I calculate VIF in R########
############################################################################
"""
Dummy-code the categorical variables to be able to assess multicollinearity
"""
# mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########

aez_classes = ["thz_class", "mst_class", "soil_class"]
aez_names = {
    "TRC_2": "Trop_high",
    "TRC_3": "Sub-trop_warm",
    "TRC_4": "Sub-trop_mod_cool",
    "TRC_5": "Sub-trop_cool",
    "TRC_6": "Temp_mod",
    "TRC_7": "Temp_cool+Bor+Arctic",
    "M_3": "120-180days",
    "M_4": "180-225days",
    "M_5": "225-270days",
    "M_6": "270+days",
    "S_2": "S2_hydro_soil",
    "S_3": "S3_no-slight_lim",
    "S_4": "S4_moderate_lim",
    "S_5": "S5_severe_lim",
    "L_3": "L3_irr",
}

data_dummy, data_correlations, spearman, Variance_inflaction = {}, {}, {}, {}
# have to decide if I want to keep the TRC,M,S classification or if I want to rename them according to the aez_names(see above)
for crop in crops:
    dummies = pd.get_dummies(
        data_clean[crop], prefix=["TRC", "M", "S"], columns=aez_classes, drop_first=True
    ).rename(columns={"S_6": "L_3"})
    data_dummy[crop] = pd.concat(
        [dummies.iloc[:, :12], data_clean[crop][aez_classes], dummies.iloc[:, 12:]],
        axis="columns",
    )  # .rename(columns=aez_names, errors='raise')
    # data_dummy[crop].to_csv(params.modelDataDir + crop + '_dummies.gzip', compression='gzip')
    data_correlations[crop] = data_dummy[crop].drop(
        [
            "lat",
            "lon",
            "area",
            "Yield",
            "n_fertilizer",
            "n_manure",
            "irrigation_rel",
            "thz_class",
            "mst_class",
            "soil_class",
            "continents",
        ],
        axis="columns",
    )
    spearman[crop] = data_correlations[crop].corr(method="spearman")
    data_correlations[crop] = add_constant(data_correlations[crop])
    Variance_inflaction[crop] = pd.Series(
        [
            variance_inflation_factor(data_correlations[crop].values, i)
            for i in range(data_correlations[crop].shape[1])
        ],
        index=data_correlations[crop].columns,
    )

variance = pd.DataFrame.from_dict(Variance_inflaction)

"""
Descriptive Statistics for each step and each crop
"""

# specify columns and apply function for all crops
metrics = ["Total_Area(ha)", "Number_Rows"]
#steps = {"raw": 0, "step1": 1, "step2": 2, "clean": 3, "outliers": 4}
steps = ["raw", "step1", "step2", "clean", "outliers"]
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
    "raw": [data_raw, columns_stat],
    "step1": [data_step1, columns_stat],
    "step2": [data_step2, columns_stat],
    "clean": [data_clean, columns_stat],
    "outliers": [outliers, factors],
}

overview_stats, df_list = {}, []
for crop in crops:
    overview_stats[crop] = [], []
    for step in steps:
        overview_stats[crop][0].append(data[step][0][crop]["area"].sum())
        overview_stats[crop][1].append(len(data[step][0][crop]))
    df = pd.DataFrame(overview_stats[crop], index=metrics, columns=steps).astype("int")
    df.columns = pd.MultiIndex.from_product([[crop], df.columns])
    df_list.append(df)
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
'''
print(desc_stats["Corn"])
descriptive_stats["Corn"]["outliers"] = pd.DataFrame.from_dict(
    out_threshold["Corn"], orient="index", columns=["Outlier_threshold"]
)
'''
# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter(params.statisticsDir + "Descriptive_Statistics.xlsx") as writer:
    # Write each dataframe to a different worksheet.
    result.to_excel(writer, sheet_name="Overview")
    desc_stats["Corn"].to_excel(writer, sheet_name="Corn")
    desc_stats["Rice"].to_excel(writer, sheet_name="Rice")
    desc_stats["Soybean"].to_excel(writer, sheet_name="Soybean")
    desc_stats["Wheat"].to_excel(writer, sheet_name="Wheat")
    #variance.to_excel(writer, sheet_name="Variance Inflation Factor")
    
"""
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

"""
