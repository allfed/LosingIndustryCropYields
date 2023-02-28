🌾📈 `LosingIndustryCropYields`
==============================

Code and notebooks for calculating crop yield in a loss of 
industry scenario. Used in ALLFED research projects.

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

## Dependencies: Setup in local dev environment (for advanced users)

Create a clone of the repository on your device.

*VERY IMPORTANT!!:*

Only *after* installing the environmnent, you need to install this repository as a pip package.

This requires installing pip. See https://pip.pypa.io/en/stable/installation/ for installation instructions.

One must run the following command in order for import commands to work between the python files (for any of the files in src/ or from scripts/ to run properly!):

```
pip install -e .
```

If any errors occur, *try re-running the command again*, this will probably fix them.

### Dependency management with Poetry

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

The pyproject.toml file lists all the dependencies if you're curious. 

### Dependency management with Anaconda

See https://docs.anaconda.com/anaconda/install/index.html for installation instructions.

Once the program is installed on your device, set up a separate environment for the project
(do not use the base environment). This step and the following can be done in two ways:
- using the GUI or
- using the Anaconda Prompt.
For people new to coding the GUI is more intuitive.

#### GUI
1. Open the Anaconda Navigator.
2. Select the tap "Environments".
3. Click "Import" and select the "loi.yml" file from the repository and name the new
    environment. All dependencies will be installed automatically.

#### Anaconda Prompt
1. Open Anaconda Prompt.
2. Type in the following line:
```bash
conda env create --name loi --file=environment.yml
```
The dependencies will be installed automatically and the environment will be name LoIYield.

This might take a few minutes.

For both versions: Code from this project will only run smoothly when opened in the new
environment and when the working directory is set to the path location of the repository on
your machine.

### Input data management

To get data for import or analysis, download from google drive:

https://drive.google.com/drive/u/1/folders/1RT73xckNdAnfDQRFiJYTcY2Yrv7HWRKs

(Shared with me -> ALLFED Research -> GIS -> Data -> LosingIndustryCropYields)

If you only want the already processed data in geopandas format, you only need 
to download the "processed" folder in.

Then, put the downloaded "raw" and/or "processed" folders in a folder called 
"data" in the LosingIndustryCropYields folder.


## How to run the code
Note: imports will lower the resolution by a large factor to increase speed of
operation! See params.ods for the resolution factor. You can set this to 1 if 
you have more than about 20 gigabytes of RAM on your machine, but it will run 
slowly at full resolution.


## What each file does

#### Notebooks

#### Python Files 

##### statistical analysis of the relationship between crop yield and its influencing factors, predicting
##### crop yields in a loss of industry scenario

src/multiple_regression/maize_highres_yield_clean.py
src/multiple_regression/rice_highres_yield_clean.py
src/multiple_regression/soybean_highres_yield_clean.py
src/multiple_regression/wheat_highres_yield_clean.py - loads yield, fertilizer, manure, pesticides, irrigation,
mechanisation and AEZ data. The data are preprocessed by eliminating missing data points
and outliers, combining categories with few observations and by checking for multicollinearity.
A generalized linear model (GLM) with gamma distribution is applied to the data to examine the
relationship between yield and the influencing factors (n_total, p_fertilizer, pesticides,
irrigation, mechanized, temperature regime, moisture regime and soil/terrain related categories).
The GLM is used to predict yields in a loss of industry scenario and the results saved as ASCII files.

src/multiple_regression/maize_highres_yield_optimized.py - as above but this file tests different resolutions.

##### importing spatial raster datasets

src/import/import_aez.py - Imports a raster (geotiff) of agroecological zones (AEZ) from the FAO at
5 arcmin resolution and saves it as a text file (.csv).

src/import/import_fertilizer.py - Imports application rates of nitrogen and phosphorus in artificial fertilizer,
upsamples it from half-degree to 5 arcmin resolution and saves the result as a text file (.csv).

src/import/import_irrigation_reliant.py - Determine irrigated area reliant on electricity 
for ground and surface water, save a text file (csv) and print summary statistics.
25 arcminute resolution. Uses aquastat and gmiav5.

src/import/upsample_irrigation.py - Upsamples the irrigation reliant csv to a 5 arcmin resolution.

src/import/import_irrigation_total.py - Imports a raster of crop irrigation area from gmiav5 at
5 arcmin resolution and saves it as a text file (.csv).

src/import/import_manure_fertilizer.py - Imports a raster (geotiff) of manure nitrogen application
rate at 5 arcmin resolution and saves it as a text file (.csv).

src/import/import_pesticide_application_bycrop.py - Imports fermanv1 pesticide data and saves 
as text (.csv) format. Additionally sums up the pesticide application of each type.

src/import/import_spam_yield_data.py - Imports crop yields and areas for each crop from 
SPAM, and saves to text file (.csv).

src/import/import_tillage.py - Imports a netcdf defined set of arrays of six classes of tillage,
reclassifying them into two new classes: 0 = Not mechanized, 1 = mechanized and saving the
result as a csv file.

##### utilities

src/utilities/params.py - Imports parameter values from the Params.ods file, including 
data directory locations, latitude and longitude grid resolution, etc.
Minimum 5 arcminute resolution.

src/utilities/plotter.py - Utility class to plot maps and countries.

src/utilities/stat_ut.py - Utility class containing useful functions involving statistical operations

src/utilities/utilities.py - Utility class containing useful functions that don't involve plotting.

src/utilities/create_asciis.py - Creates ASCII files from csv input files in order to visualize the
data resulting from the predictive analysis in QGIS.

#### misc

src/misc/Params.ods - A spreadsheet file which includes all of the custom parameters used for the model.

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


Contact: reach us at morgan@allfed.info and jessica@allfed.info.
