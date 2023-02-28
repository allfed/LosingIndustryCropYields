"""

File containing the code to prepare the input data and perform a multiple regression
on yield for maize at 5 arcmin resolution


Jessica Mörsdorf
jessica@allfed.info
jessica.m.moersdorf@umwelt.uni-giessen.de

"""

import os
import sys
import src.utilities.params as params  # get file location and varname parameters
from src.utilities.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import src.utilities.utilities as utilities

from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

# seaborn is just used for plotting, might be removed later
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
############# Data impprt and Pre-Analysis Processing#################
######################################################################


"""
Import yield data, extract zeros and plot the data
"""

# import yield data for maize

maize_yield = pd.pd.read_pickle(
    params.inputDataDir + "MAIZCropYieldHighRes.pkl", compression="zip"
)

# select all rows from maize_yield for which the column growArea has a value greater than zero
maize_nozero = maize_yield.loc[maize_yield["growArea"] > 0]
# compile yield data where area is greater 0 in a new array
maize_kgha = maize_nozero["yield_kgPerHa"]

round(np.average(maize_kgha, weights=maize_nozero["growArea"]), 2)


# sets design aspects for the following plots
matplotlib.rcParams["figure.figsize"] = (16.0, 12.0)
matplotlib.style.use("ggplot")

# plot maize yield distribution in a histogram
plt.hist(maize_kgha, bins=50)
plt.title("Maize yield ha/kg")
plt.xlabel("yield kg/ha")
plt.ylabel("density")


"""
Fitting of distributions to the data and comparing the fit
"""

# calculate loglik, AIC & BIC for each distribution
# st = stat_ut.stat_overview(dist_listm, pdf_listm, param_dictm)
#       Distribution  loglikelihood           AIC           BIC
# 7  normal on log  -7.765280e+05  1.553072e+06  1.553162e+06
# 5          Gamma  -5.184958e+06  1.036993e+07  1.037002e+07
# 1    exponential  -5.200183e+06  1.040038e+07  1.040047e+07
# 6  Inverse Gamma  -5.204101e+06  1.040822e+07  1.040831e+07
# 4     halfnormal  -5.204261e+06  1.040854e+07  1.040863e+07
# 3         normal  -5.356897e+06  1.071381e+07  1.071390e+07
# 0        lognorm  -6.250698e+06  1.250141e+07  1.250150e+07
# 2        weibull  -6.429530e+06  1.285908e+07  1.285917e+07
# best fit so far: normal on log values by far, then Gamma on non-log

"""
Import factor datasets and extract zeros,
Harmonize units and correct irrigation fraction
"""
fertilizer = pd.read_pickle(
    params.inputDataDir + "FertilizerHighRes.pkl", compression="zip"
)  # , index_col=[0])  # kg/m²
manure = pd.read_pickle(
    params.inputDataDir + "FertilizerManureHighRes.pkl", compression="zip"
)  # , index_col=[0])  # kg/km²
print("Done reading fertilizer data")
irr_t = pd.read_csv(
    params.geopandasDataDir + "FracIrrigationAreaHighRes.csv", index_col=[0]
)
crop = pd.read_pickle(
    params.inputDataDir + "FracCropAreaHighRes.pkl", compression="zip"
)  # , index_col=[0])
irr_rel = pd.read_csv(params.geopandasDataDir + "FracReliantHighRes.csv", index_col=[0])
print("Done reading irrigation data")
tillage = pd.read_pickle(
    params.inputDataDir + "TillageAllCropsHighRes.pkl", compression="zip"
)  # , index_col=[0])
aez = pd.read_pickle(
    params.inputDataDir + "AEZHighRes.pkl", compression="zip"
)  # , index_col=[0])
print("Done reading AEZ and tillage data")
continents = pd.read_pickle(params.inputDataDir + "Continents.pkl", compression="zip")
print("Done reading continent data")
m_pesticides = pd.read_pickle(
    params.inputDataDir + "CornPesticidesHighRes.pkl", compression="zip"
)


# fraction of irrigation total is of total cell area so it has to be divided by the
# fraction of crop area in a cell and set all values >1 to 1
irr_tot = irr_t["fraction"] / crop["fraction"]
irr_tot.loc[irr_tot > 1] = 1
# dividing by 0 leaves a NaN value, have to be set back to 0
irr_tot.loc[irr_tot.isna()] = 0

# fertilizer is in kg/m² and fertilizer_man is in kg/km² while yield and pesticides are in kg/ha
# all continuous variables are transformed to kg/ha
n_new = fertilizer["n"] * 10000
p_new = fertilizer["p"] * 10000
fert_new = pd.concat([n_new, p_new], axis="columns")
fert_new.rename(columns={"n": "n_kgha", "p": "p_kgha"}, inplace=True)
fertilizer = pd.concat([fertilizer, fert_new], axis="columns")  # kg/ha

applied_new = fertilizer_man["applied"] / 100
produced_new = fertilizer_man["produced"] / 100
man_new = pd.concat([applied_new, produced_new], axis="columns")
man_new.rename(
    columns={"applied": "applied_kgha", "produced": "produced_kgha"}, inplace=True
)
fertilizer_man = pd.concat([fertilizer_man, man_new], axis="columns")  # kg/ha

# compile a combined factor for N including both N from fertilizer and manure
N_total = fertilizer["n_kgha"] + fertilizer_man["applied_kgha"]  # kg/ha

"""
Loading variables into a combined dataframe and preparing the input
data for analysis by filling/eliminating missing data points, deleting
outliers and combining levels of categorical factors
"""

datam_raw = {
    "lat": maize_yield.loc[:, "lats"],
    "lon": maize_yield.loc[:, "lons"],
    "area": maize_yield.loc[:, "growArea"],
    "Y": maize_yield.loc[:, "yield_kgPerHa"],
    "n_fertilizer": fertilizer.loc[:, "n_kgha"],
    "p_fertilizer": fertilizer.loc[:, "p_kgha"],
    "n_manure": fertilizer_man.loc[:, "applied_kgha"],
    "n_man_prod": fertilizer_man.loc[:, "produced_kgha"],
    "n_total": N_total,
    "pesticides_H": m_pesticides.loc[:, "total_H"],
    "mechanized": tillage.loc[:, "is_mech"],
    "irrigation_tot": irr_tot,
    "irrigation_rel": irr_rel.loc[:, "frac_reliant"],
    "thz_class": aez.loc[:, "thz"],
    "mst_class": aez.loc[:, "mst"],
    "soil_class": aez.loc[:, "soil"],
}

# arrange data in a dataframe
dmaize_raw = pd.DataFrame(data=datam_raw)
dm0 = dmaize_raw.loc[dmaize_raw["area"] > 0]
# select only the rows where the area of the cropland is larger than 100 ha
dm0_raw = dmaize_raw.loc[dmaize_raw["area"] > 100]

dm0_raw["pesticides_H"] = dm0_raw["pesticides_H"].replace(np.nan, -9)
dm0_raw["irrigation_rel"] = dm0_raw["irrigation_rel"].replace(np.nan, 0)

# replace 0s in the moisture, temperature and soil classes as well as 7 & 8 in the
# soil class with NaN values so they can be handled with the .fillna method
dm0_raw["thz_class"] = dm0_raw["thz_class"].replace(0, np.nan)
dm0_raw["mst_class"] = dm0_raw["mst_class"].replace(0, np.nan)
dm0_raw["soil_class"] = dm0_raw["soil_class"].replace([0, 7, 8], np.nan)
# replace 8,9 & 10 with 7 in the temperature class to combine all three classes
# into one Temp,cool-Arctic class
# repalce 2 with 1 and 7 with 6 in the moisture class to compile them into one class each
dm0_raw["thz_class"] = dm0_raw["thz_class"].replace([8, 9, 10], 7)
dm0_raw["mst_class"] = dm0_raw["mst_class"].replace(2, 1)
dm0_raw["mst_class"] = dm0_raw["mst_class"].replace(7, 6)

# fill in the NaN vlaues in the dataset with a forward filling method
# (replacing NaN with the value in the preceding cell)
dm0_raw = dm0_raw.fillna(method="ffill")

# Eliminate the rows without data:
dm0_elim = dm0_raw.loc[dm0_raw["pesticides_H"] > -9]
dm0_elim = dm0_elim.loc[dm0_raw["mechanized"] > -9]
# replace remaining no data values in the fertilizer datasets with NaN and then fill them
# because there are only few left
dm0_elim.loc[dm0_elim["n_fertilizer"] < 0, "n_fertilizer"] = np.nan
dm0_elim.loc[dm0_elim["p_fertilizer"] < 0, "p_fertilizer"] = np.nan
dm0_elim = dm0_elim.fillna(method="ffill")
# replace no data values in n_total with the sum of the newly filled n_fertilizer and the
# n_manure values
dm0_elim.loc[dm0_elim["n_total"] < 0, "n_total"] = (
    dm0_elim["n_fertilizer"] + dm0_elim["n_manure"]
)

# calculate the 25th, 50th, 75th, 85th, 95th, 99th and 99.9th percentile
dm0_qt = dm0_elim.quantile([0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.999])
dm0_qt.reset_index(inplace=True, drop=True)

# Values above the 99.9th percentile are considered unreasonable outliers
# Calculate number and statistic properties of the outliers
Y_out = dm0_elim.loc[dm0_elim["Y"] > dm0_qt.iloc[6, 3]]  # ~12500
nf_out = dm0_elim.loc[dm0_elim["n_fertilizer"] > dm0_qt.iloc[6, 4]]  # ~180
pf_out = dm0_elim.loc[dm0_elim["p_fertilizer"] > dm0_qt.iloc[6, 5]]  # ~34
nm_out = dm0_elim.loc[dm0_elim["n_manure"] > dm0_qt.iloc[5, 6]]  # ~11
nt_out = dm0_elim.loc[dm0_elim["n_total"] > dm0_qt.iloc[6, 8]]  # ~195
P_out = dm0_elim.loc[dm0_elim["pesticides_H"] > dm0_qt.iloc[6, 9]]  # ~11
m_out = pd.concat(
    [
        Y_out["Y"],
        nf_out["n_fertilizer"],
        pf_out["p_fertilizer"],
        nm_out["n_manure"],
        nt_out["n_total"],
        P_out["pesticides_H"],
    ],
    axis=1,
)
m_out.max()
m_out.min()
m_out.mean()

# Eliminating all points above the 99.9th percentile
dm0_elim = dm0_elim.loc[dm0_elim["Y"] < dm0_qt.iloc[6, 3]]  # ~12500
dm0_elim = dm0_elim.loc[dm0_elim["n_fertilizer"] < dm0_qt.iloc[6, 4]]  # ~180
dm0_elim = dm0_elim.loc[dm0_elim["p_fertilizer"] < dm0_qt.iloc[6, 5]]  # ~34
dm0_elim = dm0_elim.loc[dm0_elim["n_manure"] < dm0_qt.iloc[5, 6]]  # ~11
dm0_elim = dm0_elim.loc[dm0_elim["n_total"] < dm0_qt.iloc[6, 8]]  # ~195
dm0_elim = dm0_elim.loc[dm0_elim["pesticides_H"] < dm0_qt.iloc[6, 9]]  # ~11
dm0_elim_index = dm0_elim.index

"""
Dummy-code the categorical variables to be able to assess multicollinearity
"""

# mst, thz and soil are categorical variables which need to be converted into dummy variables for calculating VIF
#####Get dummies##########
mdum_mst = pd.get_dummies(dm0_elim["mst_class"])
mdum_thz = pd.get_dummies(dm0_elim["thz_class"])
mdum_soil = pd.get_dummies(dm0_elim["soil_class"])
#####Rename Columns##########
mdum_mst = mdum_mst.rename(
    columns={
        1: "LGP<120days",
        3: "120-180days",
        4: "180-225days",
        5: "225-270days",
        6: "270+days",
    },
    errors="raise",
)
mdum_thz = mdum_thz.rename(
    columns={
        1: "Trop_low",
        2: "Trop_high",
        3: "Sub-trop_warm",
        4: "Sub-trop_mod_cool",
        5: "Sub-trop_cool",
        6: "Temp_mod",
        7: "Temp_cool+Bor+Arctic",
    },
    errors="raise",
)
mdum_soil = mdum_soil.rename(
    columns={
        1: "S1_very_steep",
        2: "S2_hydro_soil",
        3: "S3_no-slight_lim",
        4: "S4_moderate_lim",
        5: "S5_severe_lim",
        6: "L1_irr",
    },
    errors="raise",
)
# merge the dummy dataframes with the rest of the variables
dmaize_d_elim = pd.concat([dm0_elim, mdum_mst, mdum_thz, mdum_soil], axis="columns")
# drop one column of each dummy (this value will be encoded by 0 in all columns)
dmaize_dum_elim = dmaize_d_elim.drop(
    ["270+days", "Temp_cool+Bor+Arctic", "L1_irr"], axis="columns"
)

"""
Split the data into a validation and a calibration dataset
"""

# select a random sample of 20% from the dataset for validation
# random_state argument ensures that the same sample is returned each time the code is run
dmaize_val_elim = dmaize_dum_elim.sample(frac=0.2, random_state=2705)  # RAW
# drop the validation sample rows from the dataframe, leaving 80% of the data for calibrating the model
dmaize_fit_elim = dmaize_dum_elim.drop(dmaize_val_elim.index)

"""
Check for multicollinearity by calculating the two-way correlations and the VIF
"""

# extract lat, lon, area, yield, individual n columns, original climate class columns and irrigation for the LoI scenario
# from the fit dataset to test the correlations among the
# independent variables
dmaize_cor_elim = dmaize_fit_elim.drop(
    [
        "lat",
        "lon",
        "area",
        "Y",
        "n_fertilizer",
        "n_manure",
        "n_man_prod",
        "irrigation_rel",
        "thz_class",
        "mst_class",
        "soil_class",
    ],
    axis="columns",
)

#### Correlations ###

# calculate spearman (rank transformed) correlation coeficcients between the
# independent variables and save the values in a dataframe
sp_m = dmaize_cor_elim.corr(method="spearman")

### Variance inflation factor ###

Xm = add_constant(dmaize_cor_elim)
pd.Series(
    [variance_inflation_factor(Xm.values, i) for i in range(Xm.shape[1])],
    index=Xm.columns,
)

# const                40.847986
# p_fertilizer          5.509180
# n_total               6.761342
# pesticides_H          2.718888
# mechanized            1.705572
# irrigation_tot        2.207995
# LGP<120days           1.425328
# 120-180days           1.724666
# 180-225days           1.685247
# 225-270days           1.524806
# Trop_low              3.497253
# Trop_high             1.547520
# Sub-trop_warm         1.708242
# Sub-trop_mod_cool     1.683744
# Sub-trop_cool         1.349665
# Temp_mod              1.736324
# S1_very_steep         1.359520
# S2_hydro_soil         1.333951
# S3_no-slight_lim      4.021156
# S4_moderate_lim       4.026990
# S5_severe_lim         1.696426


#######################################################################
########### Regression Calibration, Validation and Residuals###########
#######################################################################


"""
Calibrate the Regression model and calculate fit statistics
"""

# determine model with a gamma distribution
m_mod_elimg = smf.glm(
    formula="Y ~ n_total + p_fertilizer + pesticides_H + irrigation_tot + mechanized + \
              C(thz_class) + C(mst_class) + C(soil_class)",
    data=dmaize_fit_elim,
    family=sm.families.Gamma(link=sm.families.links.inverse_power),
)
# Nullmodel
m_mod_elim0 = smf.glm(
    formula="Y ~ 1",
    data=dmaize_fit_elim,
    family=sm.families.Gamma(link=sm.families.links.inverse_power),
)

# Fit models
m_fit_elimg = m_mod_elimg.fit()
m_fit_elim0 = m_mod_elim0.fit()

# print results
print(m_fit_elimg.summary())
print(m_fit_elim0.summary())

# calculate the odds ratios on the response scale
np.exp(m_fit_elimg.params)

### Fit statistics ###

# calculate McFadden's roh² and the Root Mean Gamma Deviance (RMGD)
d2_tweedie_score(dmaize_fit_elim["Y"], m_fit_elimg.fittedvalues, power=2)  # 0.455
np.sqrt(
    mean_tweedie_deviance(dmaize_fit_elim["Y"], m_fit_elimg.fittedvalues, power=2)
)  # 0.5968

# calculate AIC and BIC for Gamma
m_aic = m_fit_elimg.aic
m_bic = m_fit_elimg.bic_llf
# LogLik: -2479200; AIC: 4958456; BIC: 4958677

"""
Validate the model against the validation dataset
"""

# select the independent variables from the validation dataset
m_val_elim = dmaize_val_elim.iloc[:, [5, 8, 9, 10, 11, 13, 14, 15]]

# let the model predict yield values for the validation data
m_pred_elimg = m_fit_elimg.predict(m_val_elim)

# calculate McFadden's roh² and the RMGD scores
d2_tweedie_score(dmaize_val_elim["Y"], m_pred_elimg, power=2)  # 0.4672
np.sqrt(mean_tweedie_deviance(dmaize_val_elim["Y"], m_pred_elimg, power=2))  # 0.5897

"""
Plot the Residuals for the model
"""
### Extract necessary measures ###

# select the independent variables from the fit dataset
m_fit_elim = dmaize_fit_elim.iloc[:, [5, 8, 9, 10, 11, 13, 14, 15]]

# get the influence of the GLM model
m_stat_elimg = m_fit_elimg.get_influence()

# store cook's distance in a variable
m_elimg_cook = pd.Series(m_stat_elimg.cooks_distance[0]).transpose()
m_elimg_cook = m_elimg_cook.rename("Cooks_d", errors="raise")

# store the actual yield, the fitted values on response and link scale,
# the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
# the absolute value of the resp and the sqrt of the stud residuals in a dataframe
# reset the index but keep the old one as a column in order to combine the dataframe
# with Cook's distance
m_data_infl = {
    "Yield": dmaize_fit_elim["Y"],
    "GLM_fitted": m_fit_elimg.fittedvalues,
    "Fitted_link": m_fit_elimg.predict(m_fit_elim, linear=True),
    "resid_pear": m_fit_elimg.resid_pearson,
    "resid_stud": m_stat_elimg.resid_studentized,
    "resid_resp_abs": np.abs(m_fit_elimg.resid_response),
    "resid_stud_sqrt": np.sqrt(np.abs(m_stat_elimg.resid_studentized)),
    "hat_matrix": m_stat_elimg.hat_matrix_diag,
}
m_elimg_infl = pd.DataFrame(data=m_data_infl).reset_index()
m_elimg_infl = pd.concat([m_elimg_infl, m_elimg_cook], axis="columns")


# take a sample of the influence dataframe to plot the lowess line
m_elimg_infl_sample = m_elimg_infl.sample(frac=0.01, random_state=2705)


### Studentized residuals vs. fitted values on link scale ###

# set plot characteristics
plot_ms = plt.figure(4)
plot_ms.set_figheight(8)
plot_ms.set_figwidth(12)

# Draw a scatterplot of studentized residuals vs. fitted values on the link scale
plt.scatter("Fitted_link", "resid_stud", data=m_elimg_infl)

# plot labels
plot_ms.axes[0].set_title("Studentized Residuals vs Fitted on link scale")
plot_ms.axes[0].set_xlabel("Fitted values on link scale")
plot_ms.axes[0].set_ylabel("Studentized Residuals")


### Response residuals vs. fitted values on the response scale ###

# set plot characteristics
plot_mr = plt.figure(4)
plot_mr.set_figheight(8)
plot_mr.set_figwidth(12)

# Draw a scatterplot of response residuals vs. fitted values on the response scale
plot_mr.axes[0] = sb.residplot(
    "GLM_fitted",
    "Yield",
    data=m_elimg_infl,
    # lowess=True,
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red", "lw": 1, "alpha": 0.8},
)

# plot labels
plot_mr.axes[0].set_title("Residuals vs Fitted")
plot_mr.axes[0].set_xlabel("Fitted values")
plot_mr.axes[0].set_ylabel("Residuals")

# annotations of the three largest residuals
abs_resid = m_elimg_infl["resid_resp_abs"].sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_mr.axes[0].annotate(
        i, xy=(m_elimg_infl["GLM_fitted"][i], m_elimg_infl["resid_resp_abs"][i])
    )


### QQ-Plot for the studentized residuals ###

# Specifications of the QQ Plot
QQ = ProbPlot(m_elimg_infl["resid_stud"], dist=stats.gamma, fit=True)
plot_mq = QQ.qqplot(line="45", alpha=0.5, color="#4C72B0", lw=1)

# set plot characteristics
plot_mq.set_figheight(8)
plot_mq.set_figwidth(12)

# plot labels
plot_mq.axes[0].set_title("Normal Q-Q")
plot_mq.axes[0].set_xlabel("Theoretical Quantiles")
plot_mq.axes[0].set_ylabel("Standardized Residuals")

# annotations of the three largest residuals
abs_norm_resid = np.flip(np.argsort(np.abs(m_elimg_infl["resid_stud"])), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_mq.axes[0].annotate(
        i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], m_elimg_infl["resid_stud"][i])
    )


### Cook's distance plots ###

##Cook's distance vs. no of observation##

# sort cook's distance value to get the value for the largest distance####
m_cook_sort = m_elimg_cook.sort_values(ascending=False)
# select all Cook's distance values which are greater than 4/n (n=number of datapoints)
m_cook_infl = m_elimg_cook.loc[m_elimg_cook < (4 / (156289 - 21))].sort_values(
    ascending=False
)

# barplot for values with the strongest influence (=largest Cook's distance)
# because running the function on all values takes a little longer
plt.bar(m_cook_infl.index, m_cook_infl)
plt.ylim(0, 0.01)

# plots for the ones greater than 4/n and all distance values
plt.scatter(m_cook_infl.index, m_cook_infl)
plt.scatter(m_elimg_cook.index, m_elimg_cook)
plt.ylim(0, 0.01)

##Studentized Residuals vs. Leverage w. Cook's distance line##

# set plot characteristics
plot_mc = plt.figure(4)
plot_mc.set_figheight(8)
plot_mc.set_figwidth(12)

# Draw the scatterplott of the Studentized residuals vs. leverage
plt.scatter(m_elimg_infl["hat_matrix"], m_elimg_infl["resid_stud"], alpha=0.5)
sb.regplot(
    m_elimg_infl["hat_matrix"],
    m_elimg_infl["resid_stud"],
    scatter=False,
    ci=False,
    # lowess=True,
    line_kws={"color": "red", "lw": 1, "alpha": 0.8},
)

# plot boundaries and labels
plot_mc.axes[0].set_xlim(0, 0.004)
plot_mc.axes[0].set_ylim(-3, 21)
plot_mc.axes[0].set_title("Residuals vs Leverage")
plot_mc.axes[0].set_xlabel("Leverage")
plot_mc.axes[0].set_ylabel("Standardized Residuals")

# annotate the three points with the largest Cooks distance value
leverage_top_3 = np.flip(np.argsort(m_elimg_infl["Cooks_d"]), 0)[:3]

for i in leverage_top_3:
    plot_mc.axes[0].annotate(
        i, xy=(m_elimg_infl["hat_matrix"][i], m_elimg_infl["resid_stud"][i])
    )


###########################################################################
################ Loss of Industry Modelling ###############################
###########################################################################

"""
Prepare and modify datasets according to the assumptions of the LoI scenario
"""

# take the raw dataset to calculate the distribution of remaining fertilizer/pesticides
# and available manure correctly
LoI_melim = dm0_raw

LoI_melim["mechanized"] = LoI_melim["mechanized"].replace(-9, np.nan)
LoI_melim["pesticides_H"] = LoI_melim["pesticides_H"].replace(-9, np.nan)

### Mechanised ###

# set mechanization to 0 in phase 2; due to the estimated stock in  fuel the variable remains
# unchanged in phase 1
LoI_melim["mechanized_y2"] = LoI_melim["mechanized"].replace(1, 0)

### N fertilizer ###

# drop all cells where mechanized or pesticiedes AND n_fertilizer are no data values
mn_drop = LoI_melim[
    ((LoI_melim["mechanized"].isna()) | (LoI_melim["pesticides_H"].isna()))
    & (LoI_melim["n_fertilizer"] < 0)
].index
LoI_melim_pn = LoI_melim.drop(mn_drop)

# replace remaining no data values in the fertilizer datasets with NaN and then fill them
LoI_melim_pn.loc[LoI_melim_pn["n_fertilizer"] < 0, "n_fertilizer"] = np.nan
LoI_melim_pn.loc[LoI_melim_pn["p_fertilizer"] < 0, "p_fertilizer"] = np.nan
LoI_melim_pn[["n_fertilizer", "p_fertilizer"]] = LoI_melim_pn[
    ["n_fertilizer", "p_fertilizer"]
].fillna(method="ffill")
# replace no data values in n_total with the sum of the newly filled n_fertilizer and the
# n_manure values
LoI_melim_pn.loc[LoI_melim_pn["n_total"] < 0, "n_total"] = (
    LoI_melim_pn["n_fertilizer"] + LoI_melim_pn["n_manure"]
)

# drop the outliers (99.9th percentile) in the n and p fertilizer columns
LoI_melim_pn = LoI_melim_pn.loc[LoI_melim_pn["n_fertilizer"] < dm0_qt.iloc[6, 4]]
LoI_melim_pn = LoI_melim_pn.loc[LoI_melim_pn["p_fertilizer"] < dm0_qt.iloc[6, 5]]

# in phase 1, there will probably be a slight surplus of N (production>application)
# the surplus is assumed to be the new total

# calculate kg N applied per cell
LoI_melim_pn["n_kg"] = LoI_melim_pn["n_fertilizer"] * LoI_melim_pn["area"]
# calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_pn["n_ffrac"] = LoI_melim_pn["n_kg"] / (LoI_melim_pn["n_kg"].sum())

# calculate the fraction of total N applied to maize fields of the total N applied
# divide total of maize N by 1000000 to get from kg to thousand t
m_nfert_frac = (LoI_melim_pn["n_kg"].sum()) / 1000000 / 118763
# calculate the new total for N maize in phase one based on the N total surplus
m_ntot_new = m_nfert_frac * 14477 * 1000000

# calculate the new value of N application rate in kg per ha per cell, assuming
# the distribution remains the same as before the catastrophe
LoI_melim_pn["n_fert_y1"] = (m_ntot_new * LoI_melim_pn["n_ffrac"]) / LoI_melim_pn[
    "area"
]

# in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melim_pn["n_fert_y2"] = 0

### P Fertilizer ###

# in phase 1, there will probably be a slight surplus of P (production>application)
# calculate kg p applied per cell
LoI_melim_pn["p_kg"] = LoI_melim_pn["p_fertilizer"] * LoI_melim_pn["area"]
# calculate the fraction of the total N applied to maize fields for each cell
LoI_melim_pn["p_ffrac"] = LoI_melim_pn["p_kg"] / (LoI_melim_pn["p_kg"].sum())

# calculate the fraction of total P applied to maize fields on the total P applied to cropland
# divide total of maize P by 1000000 to get from kg to thousand t
m_pfert_frac = (LoI_melim_pn["p_kg"].sum()) / 1000000 / 45858
# calculate the new total for P maize in phase one based on the P total surplus
m_ptot_new = m_pfert_frac * 4142 * 1000000

# calculate the new value of P application rate in kg per ha per cell, assuming
# the distribution remains the same as before the catastrophe
LoI_melim_pn["p_fert_y1"] = (m_ptot_new * LoI_melim_pn["p_ffrac"]) / LoI_melim_pn[
    "area"
]

# in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melim_pn["p_fert_y2"] = 0

### N Manure ###

# drop the rows containing outliers (99th percentile) in the manure column
LoI_melim_man = LoI_melim.loc[LoI_melim["n_manure"] < dm0_qt.iloc[5, 6]]

# calculate animal labor demand by dividing the area in a cell by the area a cow
# can be assumed to work
LoI_melim_man["labor"] = (
    LoI_melim_man["area"] / 5
)  # current value (7.4) is taken from Cole et al. (2016)
# due to information from a farmer, the value is set at 5

# calculate the mean excretion rate of each cow in one year: cattle supplied ~ 43.7% of 131000 thousand t
# manure production in 2014, there were ~ 1.439.413.930(FAOSTAT)
# heads of cattle in 2014
cow_excr = 131000000000 * 0.437 / 1439413930

# calculate the new value of man application rate in kg per ha per cell, according
# to the available cows in each cell according to labor demand
LoI_melim_man["man_fert"] = (cow_excr * LoI_melim_man["labor"]) / LoI_melim_man["area"]

# that leads the application rate being the same in every cell because everybody has the same number of cows per ha
# it's assumed to be the same for both phases

### N total ###

# in phase 1, the total N available is the sum of available fertilizer and manure
LoI_melim["N_toty1"] = LoI_melim_pn["n_fert_y1"] + LoI_melim_man["man_fert"]

# in phase 2 there is no more artificial fertilizer, so N total is equal to man_fert

### Pesticides ###

# drop the cells containing NaN values and outliers
LoI_melimp = LoI_melim.loc[LoI_melim["pesticides_H"].notna()]
LoI_melimp = LoI_melimp.loc[LoI_melimp["pesticides_H"] < dm0_qt.iloc[6, 9]]

# in phase 1, there will probably be a slight surplus of Pesticides (production>application)

# calculate kg p applied per cell
LoI_melimp["pest_kg"] = LoI_melimp["pesticides_H"] * LoI_melimp["area"]
# calculate the fraction of the total N applied to maize fields for each cell
LoI_melimp["pest_frac"] = LoI_melimp["pest_kg"] / (LoI_melimp["pest_kg"].sum())

# calculate the fraction of total pesticides applied to maize fields of the total pesticides applied to cropland
# divide total of maize pesticides by 1000 to get from kg to t
m_pest_frac = (LoI_melimp["pest_kg"].sum()) / 1000 / 4190985

# due to missing reasonable data on the pesticide surplus, it is assumed that the
# surplus is in the same range as for P and N fertilizer
# the mean of N and P fertilizer surplus is calculated
frac_pest = ((14477 / 118763) + (4142 / 45858)) / 2
# calculate the new total for pesticides maize in phase one based on the pesticides total surplus
m_pestot_new = m_pest_frac * (4190985 * frac_pest) * 1000

# calculate the new value of pesticides application rate in kg per ha per cell, assuming
# the distribution remains the same as before the catastrophe
LoI_melimp["pest_y1"] = (m_pestot_new * LoI_melimp["pest_frac"]) / LoI_melimp["area"]

# in phase 2 no industrially produced fertilizer will be available anymore: set to 0
LoI_melimp["pest_y2"] = 0


### Irrigation ###

# in LoI it is assumed that only irrigation which is not reliant on electricity
# can still be maintained
# calculate fraction of cropland area not reliant on electricity in a cell in LoI by multiplying
#'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
# (fraction of irrigated cropland reliant on electricity)
LoI_melim["irr_LoI"] = LoI_melim["irrigation_tot"] * (1 - LoI_melim["irrigation_rel"])


### Combine the different dataframes and drop rows with missing values ###

LoI_melim = pd.concat(
    [
        LoI_melim,
        LoI_melim_pn["n_fert_y1"],
        LoI_melim_pn["n_fert_y2"],
        LoI_melim_pn["p_fert_y1"],
        LoI_melim_pn["p_fert_y2"],
        LoI_melim_man["man_fert"],
        LoI_melimp["pest_y1"],
        LoI_melimp["pest_y2"],
    ],
    axis="columns",
)

# Eliminate the rows without data:
LoI_melim = LoI_melim.dropna()

# Eliminating all points above the 99.9th percentile

LoI_melim = LoI_melim.loc[LoI_melim["Y"] < dm0_qt.iloc[6, 3]]
LoI_melim = LoI_melim.loc[LoI_melim["n_total"] < dm0_qt.iloc[6, 8]]


"""
Prediction of LoI yields and yield change rates in phase 1 and 2
"""
### Phase 1 ###

# select the rows from LoI_melim which contain the independent variables for phase 1
LoI_m_phase1 = LoI_melim.iloc[:, [10, 13, 14, 15, 17, 19, 22, 25]]
# reorder the columns according to the order in dm0_elim
LoI_m_phase1 = LoI_m_phase1[
    [
        "p_fert_y1",
        "N_toty1",
        "pest_y1",
        "mechanized",
        "irr_LoI",
        "thz_class",
        "mst_class",
        "soil_class",
    ]
]
# rename the columns according to the names used in the model formular
LoI_m_phase1 = LoI_m_phase1.rename(
    columns={
        "p_fert_y1": "p_fertilizer",
        "N_toty1": "n_total",
        "pest_y1": "pesticides_H",
        "irr_LoI": "irrigation_tot",
    },
    errors="raise",
)

# predict the yield for phase 1 using the gamma GLM
m_yield_y1 = m_fit_elimg.predict(LoI_m_phase1)
# calculate the change rate from actual yield to the predicted yield
m_y1_change = ((m_yield_y1 - maize_kgha) / maize_kgha).dropna()
# calculate the number of cells with a postivie change rate
s1 = m_y1_change.loc[m_y1_change > 0]

# create a new variable with the yields for positive change rates set to orginial yields
m01 = m_y1_change.loc[m_y1_change > 0]
m_y1_0 = LoI_melim["Y"]
m_y1_0 = m_y1_0[m01.index]
m011 = m_y1_change.loc[m_y1_change <= 0]
m_y1_1 = m_yield_y1[m011.index]
m_y1_y0 = m_y1_0.append(m_y1_1)

# calculate statistics for yield and change rate

# calculate weights for mean change rate calculation dependent on current yield
# and current maize area in a cell
mw = LoI_melim["Y"] * dm0_elim["area"]
mw = mw.fillna(method="ffill")

# calculate weighted mean, min and max of predicted yield (1) including postive change rates
mmean_y1_weigh = round(
    np.average(m_yield_y1, weights=LoI_melim["area"]), 2
)  # 4412.24kg/ha
mmax_y1 = m_yield_y1.max()  # 12746.86 kg/ha
mmin_y1 = m_yield_y1.min()  # 675.81 kg/ha
# (2) excluding postive change rates
mmean_y1_0 = round(np.average(m_y1_y0, weights=LoI_melim["area"]), 2)  # 3357.27kg/ha
mmax_y10 = m_y1_y0.max()  # 11838.25kg/ha
mmin_y10 = m_y1_y0.min()  # 26.5kg/ha

# change rate
mmean_y1c_weigh = round(
    np.average(m_y1_change, weights=mw), 2
)  # -0.14 (~-14%) -0.2 (~-20%)
mmax_y1c = m_y1_change.max()  # +104.98 (~+10500%)
mmin_y1c = m_y1_change.min()  # -0.9048 (~-91%)

### Phase 2 ###

# select the rows from LoI_melim which contain the independent variables for phase 2
LoI_m_phase2 = LoI_melim.iloc[:, [13, 14, 15, 16, 19, 23, 24, 26]]
# reorder the columns according to the order in dm0_elim
LoI_m_phase2 = LoI_m_phase2[
    [
        "p_fert_y2",
        "man_fert",
        "pest_y2",
        "mechanized_y2",
        "irr_LoI",
        "thz_class",
        "mst_class",
        "soil_class",
    ]
]
# rename the columns according to the names used in the model formular
LoI_m_phase2 = LoI_m_phase2.rename(
    columns={
        "p_fert_y2": "p_fertilizer",
        "man_fert": "n_total",
        "pest_y2": "pesticides_H",
        "mechanized_y2": "mechanized",
        "irr_LoI": "irrigation_tot",
    },
    errors="raise",
)

# predict the yield for phase 2 using the gamma GLM
m_yield_y2 = m_fit_elimg.predict(LoI_m_phase2)
# calculate the change from actual yield to the predicted yield
m_y2_change = ((m_yield_y2 - maize_kgha) / maize_kgha).dropna()
# calculate the number of cells with a postivie change rate
s2 = m_y2_change.loc[m_y2_change > 0]

# create a new variable with all positive change rates set to 0 for both phases
m_c0 = pd.concat([m_y1_change, m_y2_change], axis=1)
m_c0 = m_c0.rename(columns={0: "m_y1_c0", 1: "m_y2_c0"}, errors="raise")
m_c0.loc[m_c0["m_y1_c0"] > 0, "m_y1_c0"] = 0
m_c0.loc[m_c0["m_y2_c0"] > 0, "m_y2_c0"] = 0


# create a new variable with the yields for positive change rates set to orginial yields
m02 = m_y2_change.loc[m_y2_change > 0]
m_y2_0 = LoI_melim["Y"]
m_y2_0 = m_y2_0[m02.index]
m022 = m_y2_change.loc[m_y2_change <= 0]
m_y2_1 = m_yield_y2[m022.index]
m_y2_y0 = m_y2_0.append(m_y2_1)

# calculate statistics for yield and change rate

# calculate weighted mean, min and max of predicted yield (1) including postive change rates
mmean_y2_weigh = round(
    np.average(m_yield_y2, weights=LoI_melim["area"]), 2
)  # 3300.81kg/ha
mmax_y2 = m_yield_y2.max()  # 8639.72kg/ha
mmin_y2 = m_yield_y2.min()  # 675.96kg/ha
# (2) excluding postive change rates
mmean_y2_0 = round(np.average(m_y2_y0, weights=LoI_melim["area"]), 2)  # 2602.48 kg/ha
mmax_y20 = m_y2_y0.max()  # 8230.62kg/ha
mmin_y20 = m_y2_y0.min()  # 26.5 kg/ha

# calculate weighted mean, min and max of predicted change rate (1) including postive change rates
mmean_y2c_weigh = round(
    np.average(m_y2_change, weights=mw), 2
)  # -0.35 (~-35%) -0.4 (~-40%)
mmax_y2c = m_y2_change.max()  # 71.02 (~+7100%)
mmin_y2c = m_y2_change.min()  # -0.9111 (~-91%)
# (2) excluding postive change rates
mmean_y2c0_weigh = round(np.average(m_c0["m_y2_c0"], weights=mw), 2)  # -0.45
mmean_y1c0_weigh = round(np.average(m_c0["m_y1_c0"], weights=mw), 2)  # -0.29


"""
Statistics to compare current SPAM2010 yield with (1) current fitted values,
(2) phase 1 and (3) phase 2 predictions
"""

## calculate statistics for current yield ##

# SPAM2010 yield: weighted mean, max, min, total yearly production
dm0_mean = round(np.average(dm0_elim["Y"], weights=dm0_elim["area"]), 2)
dm0_max = dm0_elim["Y"].max()
dm0_min = dm0_elim["Y"].min()
dm0_prod = (dm0_elim["Y"] * dm0_elim["area"]).sum()
# fitted values for current yield based on Gamma GLM: wieghted mean, max and min
m_fit_mean = round(np.average(m_fit_elimg.fittedvalues, weights=dm0_elim["area"]), 2)
m_fit_max = m_fit_elimg.fittedvalues.max()
m_fit_min = m_fit_elimg.fittedvalues.min()
m_fit_prod = (m_fit_elimg.fittedvalues * dm0_elim["area"]).sum()

## calculate statistics for both phases ##

# phase 1: calculate the percentage of current yield/production will be achieved
# in phase 1 as predicted by the GLM, calculate total production in phase 1
# (1) including positive change rates and
m_y1_per = mmean_y1_weigh / dm0_mean  # ~79.93% of current average yield
m_y1_prod = (m_yield_y1 * LoI_melim["area"]).sum()
# (2) with positive change rates set to 0
m_y10_prod = (m_y1_y0 * LoI_melim["area"]).sum()
m_y10_per = m_y10_prod / dm0_prod

# phase 2: calculate the percentage of current yield/production will be achieved
# in phase 2 as predicted by the GLM, calculate total production in phase 1
# (1) including positive change rates and
m_y2_per = mmean_y2_weigh / dm0_mean  # 59.79% of current average yield
m_y2_prod = (m_yield_y2 * LoI_melim["area"]).sum()
# (2) with positive change rates set to 0
m_y20_prod = (m_y2_y0 * LoI_melim["area"]).sum()
m_y20_per = m_y20_prod / dm0_prod

# print the relevant statistics of SPAM2010, fitted values, phase 1 and phase 2
# predictions in order to compare them
# 1st column: weighted mean
# 2nd row: total crop production in one year
# 3rd row: maximum values
# 4th row: minimum values
# last two rows comprise statistics for phase 1 and 2 (1) including positive change rates
# and (2) having them set to 0
# 5th row: percentage of current yield/production achieved in each phase
# 6th row: mean yield change rate for each phase
print(
    dm0_mean,
    m_fit_mean,
    mmean_y1_weigh,
    mmean_y2_weigh,
    dm0_prod,
    m_fit_prod,
    m_y1_prod,
    m_y2_prod,
    dm0_max,
    m_fit_max,
    mmax_y1,
    mmax_y2,
    dm0_min,
    m_fit_min,
    mmin_y1,
    mmin_y2,
    m_y1_per,
    m_y2_per,
    m_y10_per,
    m_y20_per,
    mmean_y1c_weigh,
    mmean_y2c_weigh,
    mmean_y1c0_weigh,
    mmean_y2c0_weigh,
)

"""
save the predicted yields and the yield change rates for each phase
"""

# combine yields and change rates of both phases with the latitude and longitude values
LoI_maize = pd.concat(
    [
        maize_yield["lats"],
        maize_yield["lons"],
        maize_yield["growArea"],
        m_yield_y1,
        m_y1_change,
        m_yield_y2,
        m_y2_change,
        m_c0,
    ],
    axis="columns",
)
LoI_maize = LoI_maize.rename(
    columns={0: "m_yield_y1", 1: "m_y1_change", 2: "m_yield_y2", 3: "m_y2_change"},
    errors="raise",
)

# save the dataframe in a csv
LoI_maize.to_csv(params.geopandasDataDir + "LoIMaizeYieldHighRes.csv")

# save the yield for phase 1 and 2 and the change rate for phase 1 and 2 with and without positive rates
# as ASCII files
utilities.create5minASCIIneg(
    LoI_maize, "m_y1_change", params.asciiDir + "LoIMaizeYieldChange_y1"
)
utilities.create5minASCIIneg(
    LoI_maize, "m_yield_y1", params.asciiDir + "LoIMaizeYield_y1"
)
utilities.create5minASCIIneg(
    LoI_maize, "m_y2_change", params.asciiDir + "LoIMaizeYieldChange_y2"
)
utilities.create5minASCIIneg(
    LoI_maize, "m_yield_y2", params.asciiDir + "LoIMaizeYield_y2"
)
utilities.create5minASCIIneg(
    LoI_maize, "m_y1_c0", params.asciiDir + "LoIMaizeYieldChange_0y1"
)
utilities.create5minASCIIneg(
    LoI_maize, "m_y2_c0", params.asciiDir + "LoIMaizeYieldChange_0y2"
)
