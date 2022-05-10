import geopandas as gpd
from geopandas.tools import sjoin
import pandas as pd
import numpy as np
df = pd.read_csv('../data/processed/TillageAllCropsHighRes.csv')

#15 bins of 5 arcmins.
latdiff = 1.25
londiff = 1.25

# we ignore the last latitude cell
lats = np.linspace(-90, 90 - latdiff, \
               np.floor(180 / latdiff).astype('int'))
lons = np.linspace(-180, 180 - londiff, \
               np.floor(360 / londiff).astype('int'))


lats2d, lons2d = np.meshgrid(lats, lons)

data = {"lats": pd.Series(lats2d.ravel()),
        "lons": pd.Series(lons2d.ravel())
        # "area": pd.Series(areaBinnedReoriented.ravel()),
        # "surfacewaterArea": pd.Series(swBinnedReoriented.ravel()),
        # "groundwaterArea": pd.Series(gwBinnedReoriented.ravel())
        }

df = pd.DataFrame(data=data)

geometry_plot = gpd.points_from_xy(df.lons, df.lats)
geometry_sjoin = gpd.points_from_xy(df.lons+londiff/2., df.lats+latdiff/2.)

gdf = gpd.GeoDataFrame(df, crs={'init':'epsg:4326'}, geometry=geometry_sjoin)
gdf['geometry_plot']=geometry_plot

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


#get groups of countries which match the coordinates for population losing electricity
pointInPolys = sjoin(gdf, world)

for countryIndex in set(pointInPolys.index_right.values):
    row=world[world.index==countryIndex]
    code=row['iso_a3'].values[0]
    highRes=pointInPolys[pointInPolys['iso_a3']==code]
    countryRows=aq[aq['Area'].str.startswith(code)]
    print(row)
    print(code)

