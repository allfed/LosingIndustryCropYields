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
import pandas as pd

params.importAll()

crops = {"Corn", "Rice", "Soybean", "Wheat"}
Area_Data = pd.read_csv(params.statisticsDir + "Raw_Column_Area.csv", index_col=0)

# take the cleaned dataset as a basis to calculate the conditions for the LoI scenario in phase 1 and 2
LoI_data = {}
for crop in crops:
    LoI_data[crop] = pd.read_csv(
        params.modelDataDir + crop + "_data.gzip", index_col=0, compression="gzip"
    )
    # set mechanization to 0 in phase 2; due to the estimated stock in fuel the variable remains
    # unchanged in phase 1
    LoI_data[crop]["mechanized_y2"] = 0
    # in LoI it is assumed that only irrigation which is not reliant on electricity
    # can still be maintained
    # calculate fraction of cropland area actually irrigated in a cell in LoI by multiplying
    #'irrigation_tot' (fraction of cropland irrigated in cell) with 1-'irrigation_rel'
    # (fraction of irrigated cropland reliant on electricity)
    LoI_data[crop]["irrigation_LoI"] = LoI_data[crop]["irrigation_tot"] * (
        1 - LoI_data[crop]["irrigation_rel"]
    )

### N Manure ###
# The application rate in kg/ha/cell for Loss of Industry is a constant as it is based
# on the number of cows that are needed to work a certain area. Their manure is assumed
# to be available as manure fertilisation for said area.

# calculate mean excretion rate of each cow: cattle supplied ~ 43.7% of 131000 thousand t
# manure production in 2014, there were ~ 1.439.413.930(FAOSTAT) heads of cattle in 2014
cow_excr = 131000000000 * 0.437 / 1439413930
# divide by 5, as it is assumed that one cow can work 5 ha(based on farmer info) of crop land
# (Cole et al.(2016) assume 7.4 ha)
man_const = cow_excr / 5

### Industrially produced fertilizers and pesticides ###
# in phase 1, there will probably be a small stock of fertilizers and pesticides due to production surplus
# in the previous year(production>application) -> the surplus is assumed to be the new total
# as the clean dataset eliminated many cells, dividing the new total among all cells of the clean
# dataset would leave out the need for fertilizer in the deleted cells
# to account for this, the new total will be reduced by the percentage of the area of the deleted cells


def calculate_year1(data, column, area, total):
    area_old = area[column]
    area_new = data["area"].sum()
    percentage = (area_old - area_new) / area_old
    # calculate kg substance applied per cell
    kg_cell = data[column] * data["area"]
    # calculate the fraction of the total substance applied to crop fields for each cell
    frac_cell = kg_cell / (kg_cell.sum())
    # calculate the fraction of total N applied to crop fields of the total N applied
    # divide total of crop substance by 1000 to get from kg to t
    frac_total = (kg_cell.sum()) / 1000 / total.loc[0, column]
    # calculate the new total for crop substance in phase one based on the total substance surplus
    # multiply by 1000 to get from t to kg
    # multiply with percentage to account for the missing area in the dataset
    tot_new = frac_total * total.loc[1, column] * (1 - percentage) * 1000
    # calculate the new value of substance application rate in kg per ha per cell, assuming
    # the distribution remains the same as before the catastrophe
    applied_new = (tot_new * frac_cell) / data["area"]
    return applied_new


# due to missing reasonable data on the pesticide surplus, it is assumed that the
# surplus is in the same range as for P and N fertilizer
# the mean of N and P fertilizer surplus is calculated
frac_pest = ((14477 / 118763) + (4142 / 45858)) / 2
# arrange external data in dataframe: the first value corresponds to the total yearly
# production while the second value corresponds to the yearly surplus
totals = pd.DataFrame(
    {
        "n_fertilizer": [118763000, 14477000],
        "p_fertilizer": [45858000, 4142000],
        "pesticides": [4190985, frac_pest * 4190985],
    }
)
# specify the columns for which the the new year 1 application rate will be calculated
year1_columns = ["n_fertilizer", "p_fertilizer", "pesticides"]

for crop in crops:
    LoI_data[crop]["manure_LoI"] = man_const
    # calculate the new application rate for year1_columns as specified in the function above
    for column in year1_columns:
        column_name = "{}_y1".format(column)
        LoI_data[crop][column_name] = calculate_year1(
            LoI_data[crop], column, Area_Data[crop], totals
        )
    # in phase 1, the total N available is the sum of available fertilizer and manure
    LoI_data[crop]["n_total_y1"] = (
        LoI_data[crop]["n_fertilizer_y1"] + LoI_data[crop]["manure_LoI"]
    )
    # in phase 2 there is no more artificial fertilizer, so n_fertilizer is 0 and therefore n_total equal to manure_LoI
    # in phase 2 no industrially produced fertilizers or pesticides will be available: set to 0
    LoI_data[crop][["p_fertilizer_y2", "pesticides_y2"]] = 0
    # drop all columns with data not corresponding to the LoI forcast
    LoI_data[crop] = LoI_data[crop].drop(
        [
            "Yield",
            "n_fertilizer",
            "p_fertilizer",
            "n_manure",
            "n_total",
            "pesticides",
            "irrigation_tot",
            "irrigation_rel",
        ],
        axis="columns",
    )
    # save dataframe to compressed csv file
    LoI_data[crop].to_csv(
        params.LoIDataDir + crop + "_LoI_data.gzip", compression="gzip"
    )
