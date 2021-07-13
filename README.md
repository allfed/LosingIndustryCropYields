🌾📈 `LosingIndustryCropYields`
==============================

Code and notebooks for calculating crop metrics (e.g. yield) in a loss of 
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

## Dependencies: Setup using in local dev environment (for advanced users)

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

The pyproject.toml file lists all the dependencies if you're curious. 

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

src/import_irrigation_data.py - Determine irrigated area reliant on electricity 
for ground and surface water, save a geopandas pkl and print summary statistics.
Minimum 5 arcminute resolution. Uses aquastat and gmiav5.

src/import_spam_yield.py - Imports crop yields and areas for each crop from 
SPAM, and saves to geopandas (.pkl). Minimum 5 arcminute resolution.

src/import_pesticide_application.py - Imports fermanv1 pesticide data and saves 
as (.pkl) format. Additionally sums up the pesticide application of each type.
Minimum 5 arcminute resolution.

src/import_livestock_data.py - Imports livestock head counts and saves as 
(.pkl) format. Minimum 5 arcminute resolution.

src/params.py - Imports parameter values from the Params.ods file, including 
data directory locations, latitude and longitude grid resolution, etc.
Minimum 5 arcminute resolution.

src/plotter.py - Utility class to plot maps and countries.

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