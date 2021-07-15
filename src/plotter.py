'''

A set of utility functions useful for plotting 

'''
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

import geopandas as gpd
import matplotlib.pyplot as plt
from src import params


class Plotter:
	def __init__(self):
		pass

	def plotCountryMaps(worldgdf,column,title,label,fn,show):
		worldgdf.plot(column=column,
			missing_kwds={
				"color": "lightgrey",
				"edgecolor": "red",
				"hatch": "///",
				"label": "Missing values",
			},
			legend=True,
			legend_kwds={
				'label': label,
				'orientation': "horizontal"
			}
		)
		plt.title(title)
		plt.savefig(params.figuresDir + fn + '.png')
		if(show):
			plt.show()
		else:
			plt.close()


	def plotPoints(gdf,column,title,legendlabel,fn,show):
		world=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

		fig, ax = plt.subplots(figsize=(15, 15))
		gdf.plot(ax=ax,column=column,legend=True,cmap='viridis',legend_kwds={'label': legendlabel,'orientation': "horizontal"})
		world.plot(ax=ax,facecolor='none', edgecolor='black')
		plt.savefig(params.figuresDir + fn + '.png')

		if(show):
			plt.show()

	def plotMap(df,column,title,legendlabel,fn,show):
		plt.close()
		world=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

		fig, ax = plt.subplots(figsize=(15, 15))
		df.plot(ax=ax,column=column,legend=True,cmap='viridis',legend_kwds={'label': legendlabel,'orientation': "horizontal"})
		world.plot(ax=ax,facecolor='none', edgecolor='black')
		plt.title(title)
		plt.savefig(params.figuresDir + fn + '.png')
		if(show):
			plt.show()
		else:
			plt.close()

	def mapColorByCountry(df,column,title,legendlabel,fn,show):
		plt.close()
		fig, ax = plt.subplots(figsize=(15, 15))
		df.plot(ax=ax,column=column,legend=True,cmap='viridis',legend_kwds={'label': legendlabel,'orientation': "horizontal"})
		plt.title(title)
		plt.savefig(params.figuresDir + fn + '.png')
		if(show):
			plt.show()
		else:
			plt.close()
