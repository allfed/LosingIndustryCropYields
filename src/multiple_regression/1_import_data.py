"""

File containing the code to prepare the input data and perform a multiple regression
on yield for wheat at 5 arcmin resolution


Jessica Mörsdorf
jessica@allfed.info
jessica.m.moersdorf@umwelt.uni-giessen.de

"""
import time

start_time = time.time()
import os
import sys

import src.utilities.params as params  # get file location and varname parameters
import pandas as pd
import numpy as np

params.importAll()

crops = {"MAIZ": "Corn", "RICE": "Rice", "SOYB": "Soybean", "WHEA": "Wheat"}

"""
Import yield and pesticide data for each crop
"""

######
Yield, pesticides = {}, {}
for crop in crops:
    crop_yield_name = "{}_yield".format(crop)
    Yield[crop_yield_name] = pd.read_pickle(
        params.inputDataDir + crop + "CropYieldHighRes.pkl", compression="zip"
    )
    pesticide_name = "{}_pesticides".format(crops[crop])
    pesticides[pesticide_name] = pd.read_pickle(
        params.inputDataDir + crops[crop] + "PesticidesHighRes.pkl", compression="zip"
    )
    print("Done reading " + crop_yield_name + " and " + pesticide_name + " Data")

# save only lats and lons from one yield dataframe as a compressed pickle file to use in creating the ascii files past analysis
coordinates = Yield["MAIZ_yield"][["lats", "lons"]].astype("float32")
os.makedirs(params.LoIDataDir, exist_ok=True)
coordinates.to_pickle(params.LoIDataDir + "Coordinates.pkl", compression="zip")
print("Done writing coordinates to .pkl file")

"""
Import remaining factor datasets, harmonize units and correct irrigation fraction
"""
fertilizer = pd.read_pickle(
    params.inputDataDir + "FertilizerHighRes.pkl", compression="zip"
)
manure = pd.read_pickle(
    params.inputDataDir + "FertilizerManureHighRes.pkl", compression="zip"
)
print("Done reading fertilizer data")
irr_total = pd.read_pickle(
    params.inputDataDir + "FracIrrigationAreaHighRes.pkl",
    compression="zip",
)
crop_area = pd.read_pickle(
    params.inputDataDir + "FracCropAreaHighRes.pkl", compression="zip"
)
irr_rel = pd.read_pickle(
    params.inputDataDir + "FracReliantHighRes.pkl", compression="zip"
)
print("Done reading irrigation data")
tillage = pd.read_pickle(
    params.inputDataDir + "TillageAllCropsHighRes.pkl", compression="zip"
) 
aez = pd.read_pickle(
    params.inputDataDir + "AEZHighRes.pkl", compression="zip"
)
print("Done reading AEZ and tillage data")
continents = pd.read_pickle(params.inputDataDir + "Continents.pkl", compression="zip")
print("Done reading continent data")

# fraction of irrigation total is of total cell area so it has to be divided by the
# fraction of crop area in a cell
irr_tot = irr_total["fraction"] / crop_area["fraction"]

# division by 0 or near 0 leads to NaN and Inf values -> replace with 0
irr_tot.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
# set all values >1 to 1
irr_tot.loc[irr_tot > 1] = 1

# fertilizer is in kg/m² and fertilizer_man is in kg/km² while yield and pesticides are in kg/ha
# all continuous variables are transfowmed to kg/ha
nutrients_kgha = (
    pd.DataFrame().assign(fertN_kgha=fertilizer["n"], fertP_kgha=fertilizer["p"])
    * 10000
)
nutrients_kgha[["manN_applied_kgha", "manN_produced_kgha"]] = (
    manure[["applied", "produced"]] / 100
)
# compile a combined factor for N including both N from fertilizer and manure
nutrients_kgha["totN"] = (
    nutrients_kgha["fertN_kgha"] + nutrients_kgha["manN_applied_kgha"]
)
print("Done harmonizing units and correcting irrigation fraction")

"""
Loading variables into a combined dataframe and saving the dataframe
for each crop to a compressed .pkl file
"""
data_raw, df_raw, crop_raw = {}, {}, {}
for crop in crops:
    data_name = "{}_raw".format(crops[crop])
    data_raw[data_name] = {
        "lat": Yield[crop + "_yield"].loc[:, "lats"],
        "lon": Yield[crop + "_yield"].loc[:, "lons"],
        "area": Yield[crop + "_yield"].loc[:, "growArea"],
        "Yield": Yield[crop + "_yield"].loc[:, "yield_kgPerHa"],
        "n_fertilizer": nutrients_kgha.loc[:, "fertN_kgha"],
        "p_fertilizer": nutrients_kgha.loc[:, "fertP_kgha"],
        "n_manure": nutrients_kgha.loc[:, "manN_applied_kgha"],
        # "n_man_prod": nutrients_kgha.loc[:, 'manN_produced_kgha'],
        "n_total": nutrients_kgha.loc[:, "totN"],
        "pesticides": pesticides[crops[crop] + "_pesticides"].loc[:, "total_H"],
        "irrigation_tot": irr_tot,
        "irrigation_rel": irr_rel.loc[:, "frac_reliant"],
        "mechanized": tillage.loc[:, "is_mech"],
        "thz_class": aez.loc[:, "thz"],
        "mst_class": aez.loc[:, "mst"],
        "soil_class": aez.loc[:, "soil"],
        "continents": continents.loc[:, "continent"],
    }
    df_raw[data_name] = pd.DataFrame(data=data_raw[data_name])
    crop_raw[data_name] = df_raw[data_name].loc[df_raw[data_name]["area"] > 0]
    # save raw dataframes to file to start the preprocessing with the much more lightweight
    # and much smaller cleaner .pkl files
    os.makedirs(params.cropDataDir, exist_ok=True)
    crop_raw[data_name].to_pickle(
        params.cropDataDir + data_name + ".pkl", compression="zip"
    )
    print("Done writing " + data_name + " Data to .pkl file")

end_time = time.time()
run_time = np.round(end_time - start_time, 2)
print("This file took", run_time, "seconds to complete on your system.")
