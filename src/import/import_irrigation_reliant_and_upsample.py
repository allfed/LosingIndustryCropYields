"""''
NOTE: THIS IS DESIGNED TO BE USED AT LOWER RESOLUTION (25 arcminute)
To use this, you want to set the code to 5 bins in Params.ods, run import_irrigation_reliant.py, then set it back to 1 bin and run upsample_irrigation.py.

This code imports and downsamples a raster (geotiff) of crop irrigation area
 from gmiav5 and national level government census data from aquastat.
 Imported data is estimated for 2010 from gmiav5, and usually mid-2010s for 
 aquastat.

see
https://essd.copernicus.org/articles/12/3545/2020/essd-12-3545-2020.pdf
"We prepare datafor the model based on the 2009–2011 average of the cropproduction  statistics"

Output of Import: all units are per cell in hectares
    'area' column: total irrigated area 
    'groundwaterArea' column: total irrigated area using groundwater
    'surfacewaterArea' column: total irrigated area using surfacewater
    'sw_reliant' column: total irrigated area reliant on diesel or electricity 
        using surfacewater
    'gw_reliant' column: total irrigated area reliant on diesel or electricity 
        using groundwater
    'tot_reliant' column: total irrigated area reliant on diesel or electricity


Morgan Rivers
morgan@allfed.info
7/24/21
"""
import src.utilities.utilities as utilities
import src.utilities.params as params  # get file location and varname parameters for data import
import src.utilities.plotter as Plotter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from geopandas.tools import sjoin
import os

# params.importAll()
# load the params from the params.ods file into the params object
params.importIfNotAlready()
aquastat = pd.read_csv(params.aquastatIrrigationDataLoc, index_col=False)
aq = aquastat.dropna(how="all").replace("", np.nan)

surface_irrigation_id = 4308
sprinkler_irrigation_id = 4309
localized_irrigation_id = 4310
equipped_area_id = 4311
actually_irrigated = 4461
spate_irrigation_id = 4316
lowland_irrigation_id = 4312
power_irrigation_id = 4326
total_equipped_id = 4313
flood_recession_id = 6043
ground_water_id = 4320
surface_water_id = 4321
non_equipped_flood_id = 4314
total_equipped_wetlands_id = 4315

# total area
areadata = rasterio.open(params.irrigationDataLoc + "gmia_v5_aei_ha.asc")
# hectares
gwdata = rasterio.open(params.irrigationDataLoc + "gmia_v5_aeigw_pct_aei.asc")
# surface water
swdata = rasterio.open(params.irrigationDataLoc + "gmia_v5_aeisw_pct_aei.asc")

print("reading irrigation area data")
area_array = areadata.read(1)
groundwater_array = gwdata.read(1)
surfacewater_array = swdata.read(1)
print("done reading")

# This tells us how coarsely to import irrigation (importing at very high resolution
# takes too long and doesnt help us)
AMOUNT_TO_DOWNSCALE = 5

DOWNSCALED_LATDIFF = params.latdiff * AMOUNT_TO_DOWNSCALE
DOWNSCALED_LONDIFF = params.londiff * AMOUNT_TO_DOWNSCALE

# we ignore the last latitude cell
lats = np.linspace(
    -90,
    90 - DOWNSCALED_LATDIFF,
    np.floor(180 / DOWNSCALED_LATDIFF).astype("int"),
)
lons = np.linspace(
    DOWNSCALED_LONDIFF - 180,
    180 - params.londiff * DOWNSCALED_LONDIFF,
    np.floor(360 / DOWNSCALED_LONDIFF).astype("int"),
)

latbins = np.floor(len(area_array) / len(lats)).astype("int")
lonbins = np.floor(len(area_array[0]) / len(lons)).astype("int")

area_array_resized = area_array[0 : latbins * len(lats), 0 : lonbins * len(lons)]
groundwater_array_resized = groundwater_array[
    0 : latbins * len(lats), 0 : lonbins * len(lons)
]
surfacewater_array_resized = surfacewater_array[
    0 : latbins * len(lats), 0 : lonbins * len(lons)
]
sizeArray = [len(lats), len(lons)]

# convert percent to fraction, make it zero if data < 0.
surfacewater_array_resized_fraction = (
    np.where(surfacewater_array_resized < 0, 0, surfacewater_array_resized) / 100
)
groundwater_array_resized_fraction = (
    np.where(groundwater_array_resized < 0, 0, groundwater_array_resized) / 100
)
surfacewater_array_resized_filtered = np.multiply(
    surfacewater_array_resized_fraction, area_array_resized
)
groundwater_array_resized_filtered = np.multiply(
    groundwater_array_resized_fraction, area_array_resized
)
print("rebinning data")
areaBinned = utilities.rebinCumulative(area_array_resized, sizeArray)
areaBinnedReoriented = np.fliplr(np.transpose(areaBinned))
surfacewater_binned = utilities.rebinCumulative(
    surfacewater_array_resized_filtered, sizeArray
)
swBinnedReoriented = np.fliplr(np.transpose(surfacewater_binned))
gwBinned = utilities.rebinCumulative(groundwater_array_resized_filtered, sizeArray)
gwBinnedReoriented = np.fliplr(np.transpose(gwBinned))

lats2d, lons2d = np.meshgrid(lats, lons)

print("created meshgrid")

data = {
    "lats": pd.Series(lats2d.ravel()),
    "lons": pd.Series(lons2d.ravel()),
    "area": pd.Series(areaBinnedReoriented.ravel()),
    "surfacewaterArea": pd.Series(swBinnedReoriented.ravel()),
    "groundwaterArea": pd.Series(gwBinnedReoriented.ravel()),
}

df = pd.DataFrame(data=data)
geometry_plot = gpd.points_from_xy(df.lons, df.lats)
geometry_sjoin = gpd.points_from_xy(
    df.lons + DOWNSCALED_LONDIFF / 2.0, df.lats + DOWNSCALED_LATDIFF / 2.0
)
gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry_sjoin)
gdf["geometry_plot"] = geometry_plot
# import the country boundaries, so we can see which country coordinates fall into
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# get groups of countries which match the coordinates for population losing electricity
pointInPolys = sjoin(gdf, world)

print("binned all data into polygons")


def get_value(val_id):
    vals = countryRows[countryRows["Variable Id"] == val_id]["Value"].values
    if len(vals) == 0:
        return np.nan
    else:
        return vals[0] * 1000


def estimate_reliant_scheme_sw(
    reliant_scheme,
    reliant_scheme_area,
    sw_area,
    max_area,
    spate_area,
    lowland_area,
    surface_area,
    non_equipped_flood_area,
    total_equipped_wetlands_area,
    localized_area,
    sprinkler_area,
):
    # assume lowland and spate schemes are allocated first to surface water irrigated regions

    if np.isnan(spate_area):
        spate_area_zeroed = 0
    else:
        spate_area_zeroed = spate_area
    if np.isnan(lowland_area):
        lowland_area_zeroed = 0
    else:
        lowland_area_zeroed = lowland_area
    if np.isnan(sprinkler_area):
        sprinkler_area_zeroed = 0
    else:
        sprinkler_area_zeroed = sprinkler_area
    if np.isnan(localized_area):
        localized_area_zeroed = 0
    else:
        localized_area_zeroed = localized_area
    if np.isnan(surface_area):
        surface_area_zeroed = 0
    else:
        surface_area_zeroed = surface_area
    if np.isnan(non_equipped_flood_area):
        non_equipped_flood_area_zeroed = 0
    else:
        non_equipped_flood_area_zeroed = non_equipped_flood_area
    if np.isnan(total_equipped_wetlands_area):
        total_equipped_wetlands_area_zeroed = 0
    else:
        total_equipped_wetlands_area_zeroed = total_equipped_wetlands_area

    unreliant_sw_area = (
        spate_area_zeroed
        + lowland_area_zeroed
        + non_equipped_flood_area_zeroed
        + total_equipped_wetlands_area_zeroed
    )

    # if surface water area is defined, and nonzero
    if ~np.isnan(sw_area) and sw_area > 0:
        nonsw_area = max_area - sw_area

        # remove unreliant surface water irrigation area from consideration, and assume remaining sprinkler, localized, and surface irrigation are evenly split between ground and surface water irrigated area

        if nonsw_area == 0:
            reliant_sw_area = reliant_scheme_area
            reliant_nonsw_area = 0
            reliant_sw = reliant_scheme
            reliant_nonsw = reliant_scheme
        elif unreliant_sw_area > sw_area:
            if reliant_scheme_area < nonsw_area:
                # no surface water area assigned reliant
                # reliant area all assigned to nonsw_area
                reliant_sw_area = 0
                reliant_nonsw_area = reliant_scheme_area
                reliant_sw = 0
                reliant_nonsw = reliant_nonsw_area / nonsw_area
            else:
                # this case is unlikely and hard to work with, so forget about surface water
                reliant_sw_area = reliant_scheme * sw_area
                reliant_nonsw_area = reliant_scheme * nonsw_area
                reliant_sw = reliant_scheme
                reliant_nonsw = reliant_scheme
        else:  # simple case, unreliant_sw_area < sw_area
            possibly_reliant_sw_area = sw_area - unreliant_sw_area
            possibly_reliant_nonsw_area = nonsw_area
            # now, even splitting of each, but will need to multiply out by multiplier to get back to original total fraction
            multiplier = (sw_area + nonsw_area) / (
                sw_area - unreliant_sw_area + nonsw_area
            )
            reliant_sw_area = possibly_reliant_sw_area * reliant_scheme * multiplier
            reliant_nonsw_area = (
                possibly_reliant_nonsw_area * reliant_scheme * multiplier
            )
            reliant_sw = reliant_sw_area / sw_area
            reliant_nonsw = reliant_nonsw_area / nonsw_area

        if nonsw_area > 0:
            assert (
                abs(
                    reliant_scheme
                    - (reliant_sw_area / max_area + reliant_nonsw_area / max_area)
                )
                < 0.01
            )

    else:  # surface water isn't defined, so we just use the assumed value
        reliant_nonsw = reliant_scheme
        reliant_sw = reliant_scheme

    return [reliant_sw, reliant_nonsw]


# estimate the total reliant irrigated area for the source
def estimate_source_reliance(power_area, total_irrigated_area, max_area, sw_area):
    if np.isnan(power_area):
        reliant_source = np.nan
        reliant_source_assumed = 0
        reliant_source_nonsw = reliant_source_assumed
        reliant_source_sw = reliant_source_assumed
    elif power_area == 0:
        reliant_source = 0
        reliant_source_assumed = reliant_source
        reliant_source_nonsw = reliant_source_assumed
        reliant_source_sw = reliant_source_assumed

    # we assume the source is reliant
    # if power_area is greatest area
    elif (
        np.isnan(total_irrigated_area) or total_irrigated_area <= power_area
    ) and scheme_total <= power_area:
        reliant_source = 1
        reliant_source_assumed = reliant_source
        reliant_source_nonsw = reliant_source_assumed
        reliant_source_sw = reliant_source_assumed
    else:
        reliant_source = power_area / max_area
        reliant_source_assumed = reliant_source

        # determine fraction of reliance that is likely surface water vs ground water
        # assume pumped electricity area are allocated first to non-surface water regions

        if ~np.isnan(sw_area):
            nonsw_area = max_area - sw_area
            if nonsw_area > power_area:
                reliant_source_nonsw = power_area / nonsw_area
                reliant_source_sw = 0
            else:  # nonsurface water area <= power area
                reliant_source_nonsw = 1
                reliant_source_sw = (power_area - nonsw_area) / (sw_area)
        else:  # data unavailable, don't assume difference between ground and surface water
            reliant_source_nonsw = reliant_source_assumed
            reliant_source_sw = reliant_source_assumed

    return [
        reliant_source,
        reliant_source_assumed,
        reliant_source_sw,
        reliant_source_nonsw,
    ]


def estimate_scheme_reliance(
    scheme_total,
    total_irrigated_area,
    localized_area,
    sprinkler_area,
    spate_area,
    lowland_area,
    surface_area,
    non_equipped_flood_area,
    total_equipped_wetlands_area,
    max_area,
    surface_water_area,
):
    # for now, assume total irrigated area is zero if not listed
    if np.isnan(total_irrigated_area):
        total_irrigated_area_zeroed = 0
    else:
        total_irrigated_area_zeroed = total_irrigated_area
    if np.isnan(spate_area):
        spate_area_zeroed = 0
    else:
        spate_area_zeroed = spate_area
    if np.isnan(lowland_area):
        lowland_area_zeroed = 0
    else:
        lowland_area_zeroed = lowland_area
    if np.isnan(non_equipped_flood_area):
        non_equipped_flood_area_zeroed = 0
    else:
        non_equipped_flood_area_zeroed = non_equipped_flood_area
    if np.isnan(total_equipped_wetlands_area):
        total_equipped_wetlands_area_zeroed = 0
    else:
        total_equipped_wetlands_area_zeroed = total_equipped_wetlands_area
    if np.isnan(sprinkler_area):
        sprinkler_area_zeroed = 0
    else:
        sprinkler_area_zeroed = sprinkler_area
    if np.isnan(localized_area):
        localized_area_zeroed = 0
    else:
        localized_area_zeroed = localized_area
    if np.isnan(surface_area):
        surface_area_zeroed = 0
    else:
        surface_area_zeroed = surface_area
    if np.isnan(surface_water_area):
        surface_water_area_zeroed = 0
    else:
        surface_water_area_zeroed = surface_water_area

    # catch issues with incomplete or nonsense data
    # (ignore aquastat surface and ground water area data, if data is not complete or is erroneous)
    # the scheme_total and total_irrigated area are large positive numbers of hectares, and are expected to generally be close in value to each other or equal. This conditional checks whether they are more than 1 hectare different in value.

    unreliant_scheme_irrigated = spate_area_zeroed
    +lowland_area_zeroed
    +surface_area_zeroed
    +non_equipped_flood_area_zeroed
    +total_equipped_wetlands_area_zeroed

    if abs(scheme_total) - 1 > abs(total_irrigated_area_zeroed) or abs(
        scheme_total
    ) + 1 < abs(total_irrigated_area_zeroed):
        # in this case scheme total and irrigated area total are different, so we don't have good direct estimate regarding the fraction of the scheme that is reliant.
        # if spate or lowland are listed explicitly and are nonzero: We'll guess that all non-spate and non-lowland area is electrified
        if localized_area_zeroed + sprinkler_area_zeroed == 0:
            if localized_area == 0 or sprinkler_area == 0:
                reliant_scheme = np.nan
                reliant_scheme_assumed = 0
                reliant_scheme_nonsw = reliant_scheme_assumed
                reliant_scheme_sw = reliant_scheme_assumed
            elif (  # both localized_area and sprinkler area must have been nan
                unreliant_scheme_irrigated > max_area / 2
            ):
                # reliant scheme as a fraction of area
                reliant_scheme = (max_area - unreliant_scheme_irrigated) / max_area
                reliant_scheme_assumed = reliant_scheme

                # reliant scheme as total area
                reliant_scheme_area = max_area - unreliant_scheme_irrigated

                [reliant_scheme_sw, reliant_scheme_nonsw] = estimate_reliant_scheme_sw(
                    reliant_scheme_assumed,
                    reliant_scheme_area,
                    surface_water_area,
                    max_area,
                    spate_area,
                    lowland_area,
                    surface_area,
                    non_equipped_flood_area,
                    total_equipped_wetlands_area,
                    localized_area,
                    sprinkler_area,
                )

            # we don't really know anything useful about reliant irrigation from the scheme in this condition.
            else:
                reliant_scheme = np.nan
                reliant_scheme_assumed = 0
                reliant_scheme_nonsw = reliant_scheme_assumed
                reliant_scheme_sw = reliant_scheme_assumed
        else:
            reliant_scheme = (localized_area_zeroed + sprinkler_area_zeroed) / max_area
            reliant_scheme_assumed = reliant_scheme

            reliant_scheme_area = localized_area_zeroed + sprinkler_area_zeroed
            [reliant_scheme_sw, reliant_scheme_nonsw] = estimate_reliant_scheme_sw(
                reliant_scheme_assumed,
                reliant_scheme_area,
                surface_water_area,
                max_area,
                spate_area,
                lowland_area,
                surface_area,
                non_equipped_flood_area,
                total_equipped_wetlands_area,
                localized_area,
                sprinkler_area,
            )

    # The total areas are similar. In this condition, we have good data with nonzero values, complete irrigation data, and surface water is greater than spate plus lowland irrigation
    else:
        reliant_scheme = (localized_area_zeroed + sprinkler_area_zeroed) / max_area
        reliant_scheme_assumed = reliant_scheme
        reliant_scheme_area = localized_area_zeroed + sprinkler_area_zeroed
        [reliant_scheme_sw, reliant_scheme_nonsw] = estimate_reliant_scheme_sw(
            reliant_scheme_assumed,
            reliant_scheme_area,
            surface_water_area,
            max_area,
            spate_area,
            lowland_area,
            surface_area,
            non_equipped_flood_area,
            total_equipped_wetlands_area,
            localized_area,
            sprinkler_area,
        )

    return [
        reliant_scheme,
        reliant_scheme_assumed,
        reliant_scheme_sw,
        reliant_scheme_nonsw,
    ]


# for each country,
total_gmiav5 = []
total_gmiav5_gw = []
total_gmiav5_sw = []
gmiav5_reliant_areas_sw = []
gmiav5_reliant_areas_gw = []
gmiav5_reliant_areas = []
gmiav5_all_areas = []
world["reliant"] = np.nan
world["reliant_source"] = np.nan
world["reliant_scheme"] = np.nan
world["reliant_source_sw"] = np.nan
world["reliant_scheme_sw"] = np.nan
gdf["tot_reliant"] = 0
gdf["sw_reliant"] = 0
gdf["gw_reliant"] = 0
for countryIndex in set(pointInPolys.index_right.values):
    row = world[world.index == countryIndex]
    code = row["iso_a3"].values[0]
    highRes = pointInPolys[pointInPolys["iso_a3"] == code]
    countryRows = aq[aq["Area"].str.startswith(code)]

    spate_area = get_value(spate_irrigation_id)
    lowland_area = get_value(lowland_irrigation_id)
    non_equipped_flood_area = get_value(non_equipped_flood_id)
    total_equipped_wetlands_area = get_value(total_equipped_wetlands_id)
    sprinkler_area = get_value(sprinkler_irrigation_id)
    localized_area = get_value(localized_irrigation_id)
    surface_area = get_value(surface_irrigation_id)
    power_area = get_value(power_irrigation_id)
    surface_water_area = get_value(surface_water_id)
    ground_water_area = get_value(ground_water_id)
    total_irrigated_area = get_value(total_equipped_id)

    scheme_total_arr = np.array(
        [
            surface_area,
            sprinkler_area,
            localized_area,
            spate_area,
            lowland_area,
            non_equipped_flood_area,
            total_equipped_wetlands_area,
        ]
    )

    # take the scheme areas where defined and add them up
    scheme_total_arr = scheme_total_arr[~np.isnan(scheme_total_arr)]
    scheme_total = np.sum(scheme_total_arr)

    # assign total irrigated area to the maximum of total area, scheme area, or power area, where defined
    areas = np.array([scheme_total, total_irrigated_area, power_area])
    areas = areas[~np.isnan(areas)]

    # not enough data to estimate value.
    if np.sum(areas) == 0:
        max_area = np.nan
        reliant_source = np.nan
        reliant_source_assumed = np.nan
        reliant_scheme = np.nan
        reliant_scheme_assumed = np.nan
        reliant_source_sw = reliant_source_assumed
        reliant_source_nonsw = reliant_source_assumed
        reliant_scheme_sw = reliant_scheme_assumed
        reliant_scheme_nonsw = reliant_scheme_assumed
    else:  # at least one of original areas array is not nan and not zero
        max_area = max(areas)

        [
            reliant_source,
            reliant_source_assumed,
            reliant_source_sw,
            reliant_source_nonsw,
        ] = estimate_source_reliance(
            power_area, total_irrigated_area, max_area, surface_water_area
        )

        [
            reliant_scheme,
            reliant_scheme_assumed,
            reliant_scheme_sw,
            reliant_scheme_nonsw,
        ] = estimate_scheme_reliance(
            scheme_total,
            total_irrigated_area,
            localized_area,
            sprinkler_area,
            spate_area,
            lowland_area,
            surface_area,
            non_equipped_flood_area,
            total_equipped_wetlands_area,
            max_area,
            surface_water_area,
        )

    gmiav5_all_areas.append(highRes["area"].sum())
    if np.isnan(reliant_scheme) and np.isnan(reliant_source):
        reliant = np.nan
    else:
        # the fraction not reliant is: the fraction not reliant for the scheme times the fraction not reliant for the source
        reliant = 1 - (1 - reliant_scheme_assumed) * (1 - reliant_source_assumed)
        reliant_sw = 1 - (1 - reliant_scheme_sw) * (1 - reliant_source_sw)
        reliant_nonsw = 1 - (1 - reliant_scheme_nonsw) * (1 - reliant_source_nonsw)

        world.loc[int(row.index[0]), "reliant"] = reliant
        world.loc[int(row.index[0]), "reliant_source"] = reliant_source
        world.loc[int(row.index[0]), "reliant_source_sw"] = reliant_source_sw
        world.loc[int(row.index[0]), "reliant_source_nonsw"] = reliant_source_nonsw
        world.loc[int(row.index[0]), "reliant_scheme"] = reliant_scheme
        world.loc[int(row.index[0]), "reliant_scheme_sw"] = reliant_scheme_sw
        world.loc[int(row.index[0]), "reliant_scheme_nonsw"] = reliant_scheme_nonsw

        # assert that no ground water in any country
        # is more likely than surface water to be reliant
        assert reliant_scheme_nonsw > reliant_scheme_sw - 0.001

        gmiav5_reliant_area = highRes["area"].sum() * reliant
        gmiav5_reliant_area_sw = highRes["surfacewaterArea"].sum() * reliant_sw
        gmiav5_reliant_area_gw = highRes["groundwaterArea"].sum() * reliant_nonsw

        for i, row in pointInPolys[pointInPolys["iso_a3"] == code].iterrows():
            gdf.loc[i, "tot_reliant"] = row["area"] * reliant
            gdf.loc[i, "sw_reliant"] = row["surfacewaterArea"] * reliant_sw
            gdf.loc[i, "gw_reliant"] = row["groundwaterArea"] * reliant_nonsw

        gmiav5_reliant_areas.append(gmiav5_reliant_area)
        gmiav5_reliant_areas_sw.append(gmiav5_reliant_area_sw)
        gmiav5_reliant_areas_gw.append(gmiav5_reliant_area_gw)
        total_gmiav5.append(highRes["area"].sum())
        total_gmiav5_sw.append(highRes["surfacewaterArea"].sum())
        total_gmiav5_gw.append(highRes["groundwaterArea"].sum())


print("estimated fraction irrigation where data available:")
print(np.sum(total_gmiav5) / np.sum(gmiav5_all_areas))
print("estimated year ~2010 reliant irrigation (hectares) where data available:")
print(np.sum(gmiav5_reliant_areas))
print("fraction of global irrigation that is reliant where data available:")
print(np.sum(gmiav5_reliant_areas) / np.sum(total_gmiav5))

print("fraction of global irrigation in the US that is reliant:")
print(str(float(world[world["iso_a3"] == "USA"]["reliant"]) * 100) + "%")

print("guess for fraction global surface water irrigation that is reliant:")
print(np.sum(gmiav5_reliant_areas_sw) / np.sum(total_gmiav5_sw))
print("guess for fraction global ground water irrigation that is reliant:")
print(np.sum(gmiav5_reliant_areas_gw) / np.sum(total_gmiav5_gw))

PLOT_ALL_THE_MAPS = False

if PLOT_ALL_THE_MAPS:
    title = "Fraction Overall Reliant on Electricity or Diesel"
    label = "Fraction Reliant"
    fn = "OverallReliantCountries"
    plotter = Plotter.Plotter()
    plotter.plotCountryMaps(world, "reliant", title, label, fn, True)

    title = "Fraction Scheme Sprinklers or Drip Irrigation"
    label = "Fraction Reliant"
    fn = "SchemeReliantCountries"
    plotter.plotCountryMaps(world, "reliant_scheme", title, label, fn, True)

    title = "Fraction Source Reliant on Pumps"
    label = "Fraction Reliant"
    fn = "SourceReliantCountries"
    plotter.plotCountryMaps(world, "reliant_source", title, label, fn, True)


gdf["geometry"] = gdf["geometry_plot"]
# now overlay percentage ground and surface water dependent on electricity existing irrigation area.
grid = utilities.makeGrid(gdf)
print("Saving resulting irrigation to:")
print(params.inputDataDir + "Irrigation.pkl")

os.makedirs(params.inputDataDir, exist_ok=True)
grid.to_pickle(params.inputDataDir + "Irrigation.pkl", compression="zip")

plotGrowArea = True

if PLOT_ALL_THE_MAPS:
    title = "Irrigation Area, 2005"
    label = "Area (ha)"
    plotter.plotMap(grid, "area", title, label, "IrrigationArea2005", plotGrowArea)

    title = "Surface Water Area Irrigation Area, 2005"
    label = "Area (ha)"
    plotter.plotMap(
        grid, "surfacewaterArea", title, label, "IrrigationSwArea2005", plotGrowArea
    )

    title = "Ground Water Area Irrigation Area, 2005"
    label = "Area (ha)"
    plotter.plotMap(
        grid, "groundwaterArea", title, label, "IrrigationGwArea2005", plotGrowArea
    )


print("total irrigated area: " + str(grid["area"].sum()))
print("ground water area: " + str(grid["groundwaterArea"].sum()))
print(
    "ground water area as fraction: "
    + str(grid["groundwaterArea"].sum() / grid["area"].sum())
)
print("surface water area: " + str(grid["surfacewaterArea"].sum()))
print(
    "surface water area as fraction: "
    + str(grid["surfacewaterArea"].sum() / grid["area"].sum())
)

if PLOT_ALL_THE_MAPS:
    title = "Total Irrigation reliant Area, 2005"
    label = "Area (ha)"
    plotter.plotMap(
        grid, "tot_reliant", title, label, "ReliantIrrigationArea2005", plotGrowArea
    )
    title = "Total Surface Water reliant Area, 2005"
    label = "Area (ha)"
    plotter.plotMap(
        grid, "sw_reliant", title, label, "ReliantIrrigationSwArea2005", plotGrowArea
    )
    title = "Total Ground Water reliant Area, 2005"
    label = "Area (ha)"
    plotter.plotMap(
        grid, "gw_reliant", title, label, "ReliantIrrigationGwArea2005", plotGrowArea
    )

print("Now upsampling!")


def upsample():
    MAKE_GRID = False

    mn_lat = -90
    mx_lat = 90
    mn_lon = -180
    mx_lon = 180

    # 5 arcminutes in degrees
    five_minute = 5 / 60

    pSums = {}
    nbins = 1

    start_lat_index = np.floor((90 - mx_lat) / five_minute).astype("int")
    start_lon_index = np.floor((mn_lon + 180) / five_minute).astype("int")
    # we ignore the last latitude cell
    lats = np.linspace(-90, 90 - five_minute, np.floor(180 / five_minute).astype("int"))
    lons = np.linspace(
        -180, 180 - five_minute, np.floor(360 / five_minute).astype("int")
    )

    # area_result=np.full((nbins*len(lats),nbins*len(lons)),np.nan)
    # reliant_result=np.full((nbins*len(lats),nbins*len(lons)),np.nan)
    frac_result = np.full((nbins * len(lats), nbins * len(lons)), np.nan)

    lats2d, lons2d = np.meshgrid(lats, lons)
    data = {"lats": pd.Series(lats2d.ravel()), "lons": pd.Series(lons2d.ravel())}
    df = pd.DataFrame(data=data)

    sizeArray = [len(lats), len(lons)]

    lowres_irrigation = pd.read_pickle(
        params.inputDataDir + "Irrigation.pkl", compression="zip"
    )

    # irrigated area
    area = np.array(lowres_irrigation["area"].values).astype("float32")

    # electric or diesel reliant area
    reliant = np.array(lowres_irrigation["tot_reliant"].values).astype("float32")

    # nan where not irrigated, where there's a fraction it's irrigated, and the
    # fraction is the estimated probability of that area being reliant on
    # electricity or diesel.
    lowres_fraction = np.divide(reliant, area)

    # let's make sure the input data is 10 times 5 arc minute resolution
    # which means nbins was set to 10 in the Params.ods when the irrigation data were
    # imported
    # assert(len(lowres_fraction)==93312)
    assert len(area) == 373248

    # now, we get a nice numpy array we can upsample
    # arrayWithNoData=np.where(np.bitwise_or(array<0, np.isnan(array)), -9, array)
    # flippedarr=np.ravel(np.flipud(np.transpose(lowres_fraction.reshape((4320,2160)))))
    # area_2d_lowres=area.reshape((int(2160/5),int(4320/5)))/25
    # reliant_2d_lowres=reliant.reshape((int(2160/5),int(4320/5)))/25
    frac_2d_lowres = lowres_fraction.reshape((int(2160 / 5), int(4320 / 5)))

    # print(len(area_2d_lowres))
    # print(len(area_2d_lowres[0]))

    # area_2d_highres=area_2d_lowres.repeat(5, axis=0).repeat(5, axis=1)
    # reliant_2d_highres=reliant_2d_lowres.repeat(5, axis=0).repeat(5, axis=1)
    frac_2d_highres = frac_2d_lowres.repeat(5, axis=0).repeat(5, axis=1)

    # utilities.create5minASCII(manure,'applied',params.asciiDir+'manure')
    # print(len(area_2d_highres))
    # print(len(area_2d_highres[0]))
    # print(len(area_result))
    # print(len(area_result[0]))

    # area_result[start_lat_index:start_lat_index+len(area_2d_highres),start_lon_index:start_lon_index+len(area_2d_highres[0])]=area_2d_highres
    # reliant_result[start_lat_index:start_lat_index+len(reliant_2d_highres),start_lon_index:start_lon_index+len(reliant_2d_highres[0])]=reliant_2d_highres

    frac_result[
        start_lat_index : start_lat_index + len(frac_2d_highres),
        start_lon_index : start_lon_index + len(frac_2d_highres[0]),
    ] = frac_2d_highres

    # df['area']=pd.Series((np.transpose(area_result)).ravel())
    # df['tot_reliant']=pd.Series((np.transpose(reliant_result)).ravel())

    df["frac_reliant"] = pd.Series((np.transpose(frac_result)).ravel())

    print("done upsampling")

    assert df["lats"].iloc[-1] > df["lats"].iloc[0]
    assert df["lons"].iloc[-1] > df["lons"].iloc[0]
    # quit()
    # df.sort_values(by=['lats', 'lons'],inplace=True)
    # print('2')
    # df = df.reset_index(drop=True)
    print("saving")
    os.makedirs(params.inputDataDir, exist_ok=True)
    df.to_pickle(params.inputDataDir + "FracReliantHighRes.pkl", compression="zip")


upsample()
