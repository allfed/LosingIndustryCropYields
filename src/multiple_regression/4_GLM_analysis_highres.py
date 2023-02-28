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

#######################################################################
########### Regression Calibration, Validation and Residuals###########
#######################################################################

"""
Split the data into a validation and a calibration dataset
"""

# select a random sample of 20% from the dataset to set aside for later validation
# random_state argument ensures that the same sample is returned each time the code is run
dwheat_val_elim = dwheat_duw_elim.sample(frac=0.2, random_state=2705)  # RAW
# drop the validation sample rows from the dataframe, leaving 80% of the data for fitting the model
dwheat_fit_elim = dwheat_duw_elim.drop(dwheat_val_elim.index)


"""
Calibrate the Regression model and calculate fit statistics
"""

# link = sm.families.links.log

# determine model with a gamma distribution
w_mod_elimg = smf.glm(
    formula="Y ~ n_total + p_fertilizer + irrigation_tot + mechanized + pesticides_H +  C(thz_class) + \
              C(mst_class) + C(soil_class)",
    data=dwheat_fit_elim,
    family=sm.families.Gamma(link=sm.families.links.log),
)
# Nullmodel
w_mod_elim0 = smf.glm(
    formula="Y ~ 1",
    data=dwheat_fit_elim,
    family=sm.families.Gamma(link=sm.families.links.log),
)

# Fit models
w_fit_elimg = w_mod_elimg.fit()
w_fit_elim0 = w_mod_elim0.fit()

# print results
print(w_fit_elimg.summary())
print(w_fit_elim0.summary())

# calculate the odds ratios on the response scale
coef_rs = np.exp(w_fit_elimg.params)

### Fit statistics ###

# calculate McFadden's roh² and the Root Mean Gamma Deviance (RMGD)
d2_tweedie_score(dwheat_fit_elim["Y"], w_fit_elimg.fittedvalues, power=2)
np.sqrt(mean_tweedie_deviance(dwheat_fit_elim["Y"], w_fit_elimg.fittedvalues, power=2))

# calculate AIC and BIC for Gamma
w_aic = w_fit_elimg.aic
w_bic = w_fit_elimg.bic_llf


"""
Validate the model against the validation dataset
"""

# select the independent variables from the validation dataset
w_val_elim = dwheat_val_elim.iloc[:, [5, 8, 9, 10, 11, 13, 14, 15]]

# let the model predict yield values for the validation data
w_pred_elimg = w_fit_elimg.predict(w_val_elim)
w_pred_elimg.mean()

# calculate McFadden's roh² and the RMGD scores
d2_tweedie_score(dwheat_val_elim["Y"], w_pred_elimg, power=2)
np.sqrt(mean_tweedie_deviance(dwheat_val_elim["Y"], w_pred_elimg, power=2))


"""
Plot the Residuals for the model
"""
### Extract necessary measures ###

# select the independent variables from the fit dataset
w_fit_elim = dwheat_fit_elim.iloc[:, [5, 8, 9, 10, 11, 13, 14, 15]]

# get the influence of the GLM model
w_stat_elimg = w_fit_elimg.get_influence()

# store cook's distance in a variable
w_elimg_cook = pd.Series(w_stat_elimg.cooks_distance[0]).transpose()
w_elimg_cook = w_elimg_cook.rename("Cooks_d", errors="raise")

# store the actual yield, the fitted values on response and link scale,
# the diagnole of the hat matrix (leverage), the pearson and studentized residuals,
# the absolute value of the resp and the sqrt of the stud residuals in a dataframe
# reset the index but keep the old one as a column in order to combine the dataframe
# with Cook's distance
w_data_infl = {
    "Yield": dwheat_fit_elim["Y"],
    "GLM_fitted": w_fit_elimg.fittedvalues,
    "Fitted_link": w_fit_elimg.predict(w_fit_elim, linear=True),
    "resid_pear": w_fit_elimg.resid_pearson,
    "resid_stud": w_stat_elimg.resid_studentized,
    "resid_resp_abs": np.abs(w_fit_elimg.resid_response),
    "resid_stud_sqrt": np.sqrt(np.abs(w_stat_elimg.resid_studentized)),
    "hat_matrix": w_stat_elimg.hat_matrix_diag,
}
w_elimg_infl = pd.DataFrame(data=w_data_infl).reset_index()
w_elimg_infl = pd.concat([w_elimg_infl, w_elimg_cook], axis="columns")

# take a sample of the influence dataframe to plot the lowess line
w_elimg_infl_sample = w_elimg_infl.sample(frac=0.1, random_state=2705)

### Studentized residuals vs. fitted values on link scale ###

# set plot characteristics
plot_ws = plt.figure(4)
plot_ws.set_figheight(8)
plot_ws.set_figwidth(12)

# Draw a scatterplot of studentized residuals vs. fitted values on the link scale
# lowess=True draws a fitted line which shows if the relationship is linear
plot_ws.axes[0] = sb.regplot(
    "Fitted_link",
    "resid_stud",
    data=w_elimg_infl_sample,
    lowess=True,
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red", "lw": 1, "alpha": 0.8},
)
# plt.scatter('Fitted_link', 'resid_stud', data=w_elimg_infl)

# plot labels
plot_ws.axes[0].set_title("Studentized Residuals vs Fitted on link scale")
plot_ws.axes[0].set_xlabel("Fitted values on link scale")
plot_ws.axes[0].set_ylabel("Studentized Residuals")

### Response residuals vs. fitted values on the response scale ###

# set plot characteristics
plot_wr = plt.figure(4)
plot_wr.set_figheight(8)
plot_wr.set_figwidth(12)

# Draw a scatterplot of response residuals vs. fitted values on the response scale
plot_wr.axes[0] = sb.residplot(
    "GLM_fitted",
    "Yield",
    data=w_elimg_infl_sample,
    lowess=True,
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red", "lw": 1, "alpha": 0.8},
)

# plot labels
plot_wr.axes[0].set_title("Residuals vs Fitted")
plot_wr.axes[0].set_xlabel("Fitted values")
plot_wr.axes[0].set_ylabel("Residuals")

# annotations of the three largest residuals
abs_resid = w_elimg_infl_sample["resid_resp_abs"].sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_wr.axes[0].annotate(
        i,
        xy=(
            w_elimg_infl_sample["GLM_fitted"][i],
            w_elimg_infl_sample["resid_resp_abs"][i],
        ),
    )

### QQ-Plot for the studentized residuals ###

# Specifications of the QQ Plot
QQ = ProbPlot(w_elimg_infl["resid_stud"], dist=stats.gamma, fit=True)
plot_wq = QQ.qqplot(line="45", alpha=0.5, color="#4C72B0", lw=1)

# set plot characteristics
plot_wq.set_figheight(8)
plot_wq.set_figwidth(12)

# plot labels
plot_wq.axes[0].set_title("Normal Q-Q")
plot_wq.axes[0].set_xlabel("Theoretical Quantiles")
plot_wq.axes[0].set_ylabel("Standardized Residuals")

# annotations of the three largest residuals
abs_norm_resid = np.flip(np.argsort(np.abs(w_elimg_infl["resid_stud"])), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_wq.axes[0].annotate(
        i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], w_elimg_infl["resid_stud"][i])
    )

### Cook's distance plots ###

##Cook's distance vs. no of observation##

# sort cook's distance value to get the value for the largest distance####
w_cook_sort = w_elimg_cook.sort_values(ascending=False)
# select all Cook's distance values which are greater than 4/n (n=number of datapoints)
w_cook_infl = w_elimg_cook.loc[w_elimg_cook > (4 / (168227 - 21))].sort_values(
    ascending=False
)

# barplot for values with the strongest influence (=largest Cook's distance)
# because running the function on all values takes a little longer
plt.bar(w_cook_infl.index, w_cook_infl)
plt.ylim(0, 0.01)

# plots for the ones greater than 4/n and all distance values
plt.scatter(w_cook_infl.index, w_cook_infl)
plt.scatter(w_elimg_cook.index, w_elimg_cook)
plt.ylim(0, 0.01)

##Studentized Residuals vs. Leverage w. Cook's distance line##

# set plot characteristics
plot_wc = plt.figure(4)
plot_wc.set_figheight(8)
plot_wc.set_figwidth(12)

# Draw the scatterplott of the Studentized residuals vs. leverage
plt.scatter(
    w_elimg_infl_sample["hat_matrix"], w_elimg_infl_sample["resid_stud"], alpha=0.5
)
sb.regplot(
    w_elimg_infl_sample["hat_matrix"],
    w_elimg_infl_sample["resid_stud"],
    scatter=False,
    ci=False,
    lowess=True,
    line_kws={"color": "red", "lw": 1, "alpha": 0.8},
)

# plot boundaries and labels
plot_wc.axes[0].set_xlim(0, 0.004)
plot_wc.axes[0].set_ylim(-3, 21)
plot_wc.axes[0].set_title("Residuals vs Leverage")
plot_wc.axes[0].set_xlabel("Leverage")
plot_wc.axes[0].set_ylabel("Standardized Residuals")

# annotate the three points with the largest Cooks distance value
leverage_top_3 = np.flip(np.argsort(w_elimg_infl_sample["Cooks_d"]), 0)[:3]

for i in leverage_top_3.index:
    plot_wc.axes[0].annotate(
        i,
        xy=(w_elimg_infl_sample["hat_matrix"][i], w_elimg_infl_sample["resid_stud"][i]),
    )

"""
Prediction of LoI yields and yield change rates in phase 1 and 2
"""
### Phase 1 ###

# select the rows from LoI_relim which contain the independent variables for phase 1
LoI_w_phase1 = LoI_welim.iloc[:, [10, 13, 14, 15, 23, 27, 30, 32]]
# reorder the columns according to the order in dw0_elim
LoI_w_phase1 = LoI_w_phase1[
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
LoI_w_phase1 = LoI_w_phase1.rename(
    columns={
        "p_fert_y1": "p_fertilizer",
        "N_toty1": "n_total",
        "pest_y1": "pesticides_H",
        "irr_LoI": "irrigation_tot",
    },
    errors="raise",
)

# predict the yield for phase 1 using the gamma GLM
w_yield_y1 = w_fit_elimg.predict(LoI_w_phase1)


# Code to calculate LoI predictions 95% confidence interval

# Calculate the confidence intervals for the predicted mean on the response scale
w_r_y1 = w_fit_elimg.get_prediction(LoI_w_phase1)
w_r_y1_conf = w_r_y1.summary_frame(alpha=0.05)
# verify that the results are the same as with the other predict method
np.mean(np.isclose(w_r_y1_conf.iloc[:, 0], w_yield_y1))
# calculate the confidence and prediction intervals for the predicted mean on the link scale
w_l_y1 = w_r_y1.linpred
w_l_y1_cp = w_l_y1.summary_frame()


# calculate the change rate from actual yield to the predicted yield
w_y1_change = ((w_yield_y1 - wheat_kgha) / wheat_kgha).dropna()
# calculate the number of cells with a postivie change rate
s1 = w_y1_change.loc[w_y1_change > 0]

# create a new variable with the yields where increased yields are set to orginial yields
w01 = w_y1_change.loc[w_y1_change > 0]
w_y1_0 = LoI_welim["Y"]
w_y1_0 = w_y1_0[w01.index]
w011 = w_y1_change.loc[w_y1_change <= 0]
w_y1_1 = w_yield_y1[w011.index]
w_y1_y0 = w_y1_0.append(w_y1_1)

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

### Phase 2 ###

# select the rows from LoI_welim which contain the independent variables for phase 2
LoI_w_phase2 = LoI_welim.iloc[:, [13, 14, 15, 16, 24, 26, 31, 32]]
# reorder the columns according to the order in dw0_elim
LoI_w_phase2 = LoI_w_phase2[
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
LoI_w_phase2 = LoI_w_phase2.rename(
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
w_yield_y2 = w_fit_elimg.predict(LoI_w_phase2)
# calculate the change from actual yield to the predicted yield
w_y2_change = ((w_yield_y2 - wheat_kgha) / wheat_kgha).dropna()
# calculate the number of cells with a postivie change rate
s2 = w_y2_change.loc[w_y2_change > 0]

# create a new variable with all positive change rates set to 0 for both phases
w_c0 = pd.concat([w_y1_change, w_y2_change], axis=1)
w_c0 = w_c0.rename(columns={0: "w_y1_c0", 1: "w_y2_c0"}, errors="raise")
w_c0.loc[w_c0["w_y1_c0"] > 0, "w_y1_c0"] = 0
w_c0.loc[w_c0["w_y2_c0"] > 0, "w_y2_c0"] = 0

# create a new variable with the yields where increased yields are set to orginial yields
w02 = w_y2_change.loc[w_y2_change > 0]
w_y2_0 = LoI_welim["Y"]
w_y2_0 = w_y2_0[w02.index]
w022 = w_y2_change.loc[w_y2_change <= 0]
w_y2_1 = w_yield_y2[w022.index]
w_y2_y0 = w_y2_0.append(w_y2_1)

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

continents = pd.read_csv(params.geopandasDataDir + "Continents.csv")
cont = continents.iloc[w_yield_y1.index]
t = pd.concat(
    [cont, LoI_welim], axis=1, join="inner"
)  # sanity check to see if coordinates match
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
# zon_stat['continent'] = zon_stat['continent'].replace(0, np.nan)
# zon_stat = zon_stat.fillna(method = 'ffill')

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


zon_p = pd.concat(
    [
        zon_stat["w_yield_y1"],
        zon_stat["w_yield_y2"],
        zon_stat["w_y1_change"],
        zon_stat["w_y2_change"],
    ],
    axis=1,
)
wp = zon_stat["area"]

wm_p = stat_ut.weighted_mean_zonal(zon_p, new_cont, wp)


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
