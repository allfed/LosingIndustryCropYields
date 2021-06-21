'''

A set of utility functions useful for plotting 

'''

import geopandas as gpd
import matplotlib.pyplot as plt
from src import params


class Plotter:
	def __init__(self):
		pass

	def plotPoints(gdf,column,title,legendlabel,fn,show):
		world=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

		fig, ax = plt.subplots(figsize=(15, 15))
		gdf.plot(ax=ax,column=column,legend=True,cmap='viridis',legend_kwds={'label': legendlabel,'orientation': "horizontal"})
		world.plot(ax=ax,facecolor='none', edgecolor='black')
		plt.savefig("../" + params.figuresDir + fn + '.png')

		if(show):
			plt.show()

	def plotMap(grid,column,title,legendlabel,fn,show):
		plt.close()
		world=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

		fig, ax = plt.subplots(figsize=(15, 15))
		grid.plot(ax=ax,column=column,legend=True,cmap='viridis',legend_kwds={'label': legendlabel,'orientation': "horizontal"})
		world.plot(ax=ax,facecolor='none', edgecolor='black')
		plt.title(title)
		plt.savefig("../" + params.figuresDir + fn + '.png')
		if(show):
			plt.show()

