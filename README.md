🌾📈 `CropOpt`
==============================

Code and notebooks for calculating crop metrics (e.g. yield) in a nuclear
winter scenario. Used in ALLFED research projects. 

Code can be run either in the interactive Colab environment (in which case
you don't need to clone this repo locally), or locally using Python. 
Setup instructions for both are below.

# Setup

## Setup using Colab

In GitHub, navigate to the `notebooks` folder and click on the notebook you
want to run. Then, click the "Open in Colab" button at the top of the notebook.
This will open the notebook using Colab - just follow the instructions in 
the notebook. For anyone new to Colab, you can go through an intro demo here:
https://colab.research.google.com/notebooks/basic_features_overview.ipynb

See "what each file does" section below for an introduction to each notebook.

## Setup using a local dev environment (only recommended for advanced users)

Dependency management is done with Poetry.
See https://python-poetry.org/docs/ for installation instructions.

Once it's installed, in the root folder of the repo, run:

```bash
poetry install
```

The easiest way to run the code is by activating a virtual environment for this
project using Poetry:

```bash
poetry shell
````

To exit the shell, simply type:
```bash
exit
```

The pyproject.toml file lists all the dependenciesm if you're curious. 

# How to use

## Summary of what the code is doing

Code will loop through precipitation, temperature, and rainfall data, save images of the plots, and optionally plot the maps.

Note I am not an agricultural economist, and so the predictions are not at all 
accurate yet. The purpose of the code so far is to demonstrate the GIS aspect, without attempting realistic modelling. Several things which we will likely
need to change:
 - uses 'ideal growth' from the google sheets and modifies this by rain and 
 temperature growth coefficients, rather than using real world crop outputs
 - uses all available cropland for each crop. Realistic models would not use 
 all the crop area for each crop, but would use some fraction for each
 - only has spring wheat and spring barley
 - the attempt at finding the growing season for the crop may need to be 
 removed or redone. Right now it simply finds the highest average temperature 
 of the region for the months of the growing seasons, which is probably not 
 the best way to estimate growing season for each crop
 - there are mysterious zeros in the data for some months, which I believe has 
 to do with incorrect temperature importing

The data used for plotting and geospatial analysis is in the geopandas format 
(.pkl).

RawNuclearWinterData is not included, as it's several gigabytes. If importing 
directly from Toon group nuclear winter data, netCDF4 is also required. If 
downloading from Toon group directly, you will want to place the data in a new 
folder called "RawNuclearWinterData" in the Data/ directory


## How to run the code
Usage: 

Once the dependencies have been installed (see Dependencies section of this 
readme), run the command:

$ jupyter notebook CropOpt_demo.ipynb

The jupyter notebook should show up in your default browser. You should then select "Restart & Run All" from the kernel menu item.

If you've downloaded all the Toon group data in the RawNuclearWinterModel (you 
would need to have been given access to this) you can also run the following 
command:

$ jupyter notebook Import_demo.ipynb

To set parameters, you need to modify the Params.ods and update Params.py 
accordingly. Each new variable needs to be assigned as a global inside of the 
appropriate function corresponding to the Params.ods sheet. The conditional 
values also have to be set.

After the plotting, a CSV of the temperature vs crop area is saved in Data/.

## What each file does

#### Notebooks

yield_demo.ipynb - The main program. Runs Modules/OutdoorGrowth.py to 
calculate yield. Imports from ImportedAsGeopandas/ .pkl files. Runs through 
the plots, showing precipitation, solar flux, specific humidity, and 
temperature for each month. Written in an interactive jupyter notebook.

import_demo.ipynb - Runs the Import scripts in an interactive jupyter 
notebook. Lists available variables from Toon group (Robock) dataset.

#### Python Files 

src/import_atm_data.py - Imports raw nuclear winter atmospheric data from NetCDF
format, and saves to geopandas (.pkl).

src/import_clm_data.py - Imports raw nuclear winter land data from NetCDF 
format, and saves to geopandas (.pkl).

src/import_grow_area_data.py - Imports arable land data from the FAO into 
(.pkl) format

src/params.py - Imports parameter values from the Params.ods file, including 
data directory locations, latitude and longitude grid resolution, and which 
variables to import to .pkl files from the raw nuclear winter data.

src/plotter.py - Utility class to plot maps and countries.

src/outdoor_growth.py - Calculates yield from outdoor growth.

#### misc

Params.ods - A spreadsheet file which includes all of the custom parameters used for the model.

Project Organization
------------

    ├── README.md        <- The top-level README for developers using this project.
    ├── data
    │   ├── processed    <- The final, canonical data sets for modeling.
    │   └── raw          <- The original, immutable data dump.
    │
    ├── notebooks        <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                       the creator's initials, and a short `-` delimited description, e.g.
    │                       `1.0-jqp-initial-data-exploration`.
    │
    ├── reports          <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── literature   <- Relevant papers.
    │   └── figures      <- Generated graphics and figures to be used in reporting.
    │
    ├── pyproject.toml   <- Contains all the python packages used for poetry. 
    ├── poetry.lock      <- File showing exact version of dependencies for reproducing the analysis environment.
    ├── src              <- Source code for use in this project.
        └── modules      <- Python algorithms specific to each crop.

--------


Contact: reach me at morgan@allfed.info.