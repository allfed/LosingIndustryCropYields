🌾📈 `LosingIndustryCropYields`
==============================

Code and notebooks for calculating crop yield in a loss of industry scenario.
A generalized linear model based on a gamma distribution is fitted to global
spatial yield data for corn, rice, soybean and wheat. Predictors are data on
the nitrogen application rate in kg/ha, the pesticide application rate in kg/ha,
the fraction of irrigated crop land, if agricultural work is executed with machinery
or not and three climate classes to control for climatic yield differences.
The model predicts crop yields in two phases following a global catastrophe
which inhibits the usage of any electric services. Phase 1 reflects conditions
in the year immediately  after the catastrophe, assuming the availability of
fertilizer, pesticides, and fuel stocks. However, those stocks would be subject to
rationed use in the first year. In phase 2, all stocks are used up and artificial fertilizer,
pesticides and fuel are not available anymore. This work provides a first crop-specific
and spatially explicit estimate on how strongly yields could be affected by a
catastrophic scenario which inhibits global industry. The general trends visible
in the prediction results (see reports/figures) are reliable and can be used as
a guideline going forward. However, it is not recommended to use the generated
datasets in regional analysis or for detailed response planning. Achieving the
necessary level of model accuracy for these applications was beyond the scope of this work.
Used in ALLFED research projects.

Code can be run locally using Python (and optionally R). Setup instructions are below.

The code does not have to be run to look at the results. All outputs of the code can
be found in the reports and the processed/output folders. The reports folder also contains
detailed documentation on the report files.

# Setup

## Dependencies: Setup in local dev environment

Create a clone of the repository on your device.

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

The dependencies will be installed automatically and the environment will be named LoIYield.

This might take a few minutes.

*VERY IMPORTANT!!:*

Only *after* installing the environmnent, you need to install this repository as a pip package.

This requires installing pip. See https://pip.pypa.io/en/stable/installation/ for installation instructions.

One must run the following command in order for import commands to work between the python files (for any of the files in src/ or from scripts/ to run properly!):

```
pip install -e .
```

If any errors occur, *try re-running the command again*, this will probably fix them.

### Set up instructions for R:

*Installing R is necessary to perform a full statistical analysis of the results, but can be skipped if
you are not interested in running the code for the variance inflation factor and the residual plots.
The output of this code can be found in reports/figures (residual plots) and the sheet Model_VIF
in the reports/Model_results.xlsx.*

You may download R here: https://cran.rstudio.com/

Download and install R before installing RStudio.

You may download RStudio here: https://posit.co/download/rstudio-desktop/

Open the R project file LosingIndustryCropYield.Rproj from your local clone of the repository.
From here you can open and run 4.2_Residuals+VIF.R.

### Input data management

The raw data is only necessary if you want to run the run_all_imports.py script. The
run_analysis.py script only takes files from the “processed/input” folder as inputs.

If you want to do this you can either refer to Table 1 in the corresponding paper and
download the data from their original sources or you can contact us at morgan@allfed.info
and we can share a compiled data folder with you.

Once you have downloaded the data, create a “data” folder in the root folder of the repository
and save the downloaded "raw” folder to “data”. 

For more information on the input data refer to Description_input_data.pdf.

## How to run the code
Note: imports will lower the resolution by a large factor to increase speed of
operation! See params.ods for the resolution factor. You can set this to 1 if 
you have more than about 20 gigabytes of RAM on your machine, but it will run 
slowly at full resolution.

Contact: reach us at morgan@allfed.info or jessi.moersdorf@gmail.com

# Files

## Project structure

```
├── data
├── processed
│   ├── input
│   │   ├── crop
│   │   ├── LoI
│   │   ├── model
│   │   └── raw
│   └── output
├── reports
│   └── figures
├── scripts
│   ├── run_all_imports.py
│   └── run_analysis.py
├── setup.py
├── spatial_data
│   ├── countries
│   ├── oceans
│   └── regions
└── src
├── import
	│   ├── import_aez.py
 	│   ├── import_cell_area_data.py
  	│   ├── import_continents.py
   	│   ├── import_crop_area_data.py
    	│   ├── import_fertilizer.py
     	│   ├── import_irrigation_reliant_and_upsample.py
      	│   ├── import_irrigation_total.py
       	│   ├── import_manure_fertilizer.py
	│   ├── import_pesticide_application_bycrop.py
 	│   ├── import_spam_yield_data.py
  	│   └── import_tillage.py
   	├── misc
    	├── multiple_regression
     	│   ├── 1_import_data.py
      	│   ├── 2_data_preprocessing.py
       	│   ├── 3_LoI_scenario_data.py
	│   └── 4.1_GLM_analysis.py
 	└── utilities
  	    ├── params.py
       	    ├── plotter.py
	    ├── stat_ut.py
     	    └── utilities.py
```

## Python function calls

All python file calls within the repository are shown below. The dotted lines indicate definitions of
objects, while the solid lines indicate the calling of instantiated objects. Files without function calls or 
object instantiation are not shown. The diagram above was created with pyan3 (https://github.com/Technologicat/pyan).

![output](https://github.com/allfed/LosingIndustryCropYields/assets/66425056/01a37879-feff-4750-815a-36d5be2b860d)

## Which file does what?

### Data

This folder contains the input data. Due to the size of the input files, it is not part of 
the repository. To get the data, consult the information in the section “Input data management”
or refer to Table 1 in the corresponding paper to retrieve the data from their original sources.

### Processed

This folder contains intermediary and output files of the analysis.

#### Input

This folder contains intermediary files generated in different steps of the data preparation
process and used in the course of the analysis.

**Crop**: Contains one pickle file for each crop encompassing all relevant variables for the analysis.
Output from src/multiple_regression/1_import_data.py.

**LoI**: Contains one zipped csv file that holds the input data for the loss of industry predictions
for each crop as well as a pickle file containing the coordinates of the whole world to create global
ascii files in src/multiple_regression/4.1_GLM_analysis. Output from src/multiple_regression/3_LoI_scenario_data.py.

**Model**: Contains one zipped csv file per crop. It holds the cleaned input data for the model
calibration. Output from src/multiple_regression/2_data_preprocessing.py. Contains one index file
for each crop which is used in src/multiple_regression/4.2_Residuals+VIF.R.

**Raw**: Contains pickle files for each input data set. Output from src/import folder.

#### Output

This folder contains 12 ascii result files per crop which are generated by
src/multiple_regression/4.1_GLM_analysis.py. They include predictions of yield and yield reduction
for phase 1 and 2 in a loss of industry scenario. For each crop, phase and metric the predicted mean
and the high and low bounds of the 95% confidence interval are presented.

### Reports

For more information on the stats and figures illustrated in the reports refer to reports/reports_descritprion.pdf.
The folder holds excel files containing descriptive statistics and model results.

#### Figures

This folder contains residual plots, maps showing the spatial distribution of the predicted yield
reduction for each crop in both phases, a plot showing the distribution of yield reduction across
continents and a plot displaying the average yield reduction for each crop by phase.

### Scripts

**Reduction_by_continent.ipynb**: Jupyter notebook to create a plot showing the predicted yield
reduction for each crop and phase by continent. The sheet Continent_statistics from the file
reports/Prediction_statistics.xlsx is taken as input for the plot. The output is saved to the
reports/figures folder. For more information on the output refer to reports/reports_descritprion.docx.

**Reduction_by_crop.ipynb**: Jupyter notebook to create a plot showing the predicted yield reduction
for each crop and phase. The sheet Prediction_statistics from the file reports/Prediction_statistics.xlsx
is taken as input for the plot. The output is saved to the reports/figures folder.
For more information on the output refer to reports/reports_descritprion.docx

**Spatial Distribution Plots.ipynb**: Jupyter notebook to create one map for each crop and one map for
each phase which show the spatial distribution of the predicted yield reduction. The results are saved
to the reports/figures folder. For more information on the output refer to reports/reports_descritprion.docx.

**Run_all_imports.py**: Runs all code from the src/import folder.

**Run_analysis.py**: Runs all Python code from the src/multiple_regression folder.

### Spatial_data

This folder contains spatial data used in the Reduction_by_continent.ipynb and
the Spatial Distribution Plots.ipynb scripts.

### Src

This folder contains all of the relevant code for importing, preparing, analyzing and modeling the data.

**__init__.py**: ensures that the repository can be installed as a package.

#### import

This folder contains code to import the raw input datasets from the data folder and convert them into
tables saved in the pickle format to the processed/input/raw folder.

#### misc

**Params.ods**: Open document file which contains the file paths for the variables used in the process
of preparing and executing the analysis.

#### multiple_regression

This folder contains the code for preparing the data and executing the analysis. For more details on
the process or the input and output files of each .py file, refer to the descriptions at the top of
each code file.

**1_import_data.py**: This code loads all raw input data sets (yield, pesticides, AEZ classes,
fertilizer, manure, irrigation, crop area, irrigation fraction based on fossil fuels, tillage)
generated by the code in the import folder, harmonizes the units and creates one data frame for
each crop containing all relevant data. The input data frame for each crop is then saved to the
processed/input/crop folder in the pickle format. Also saves the world coordinates at 5 arcmin
resolution to the processed/input/LoI folder for use in 3_LoI_scenario_data.py.

**2_data_preprocessing.py**: This code takes the output of 1_import_data.py as input. The first
part of the code prepares the input data for the following GLM analysis. The cleaned data sets are
saved to the processed/input/model folder in a zipped csv format. It also calculates the total crop
area and saves it to the reports folder in csv format for use in the next step.

The second part calculates descriptive statistics for each step of the data cleaning process to
compare the impact each step has on the data. The statistics are saved as excel files to the
reports folder (Descriptive_statistics.xlsx).

**3_LoI_scenario_data.py**: This code loads the clean crop data sets (output of 2_data_preprocessing.py)
which represent the conditions of the present day. Taking this data as well as assumptions and data from
the FAO as the basis, the conditions for the phases of the loss of industry scenario are simulated.
The loss of industry data sets are saved to the processed/input/LoI folder in a zipped csv format.

**4.1_GLM_analysis.py**: The first part of this code prepares and executes the fitting of a generalized
linear model based on a gamma distribution with a log link to the four cleaned crop data sets (output of
2_data_preprocessing). It calculates and saves model results and statistics to the reports folder
Model_results.xlsx).

The second part applies the model to predict the yield under loss of industry circumstances in two phases
(see data preparation in 3_LoI_scenario_data). The predicted yields and yield reductions are saved to the
processed/output folder as ascii files. Descriptive statistics are calculated to compare yield, yield
loss and yearly production between a) the current situation, the model results for the current situation
(fitted values), phase 1 and phase 2 of the loss of industry scenario and b) the continents.
All statistics are saved to the reports folder as an excel file (Prediction_statistics.xlsx).

**4.2_Residuals+VIF.R**: The first part of the code loads the crop data generated by 2_data_preprocessing.py,
fits the generalized linear model with gamma distribution and log link and calculates the generalized
variance inflation factor for each crop. It saves the GVIF to an existing excel file in the reports folder
(Model_results.xlsx).

The second part creates two types of residual plots for each crop in the reports folder: 
a) A single scatterplot showing the studentized residuals plotted against the fitted values on the link scale.
b) Four plots based on the standardized deviance residuals and the fitted values on the response scale:
Residuals vs. Fitted, Q-Q-Plot, Location-Scale Plot, Residual-Leverage Plot.
For more information refer to reports/report_descriptions.docx

#### utilities

This folder contains sets of functions which are called from different locations in the code.

**params.py**: Code to import file paths specified in misc/params.ods as globals.

**plotter.py**: A set of utility functions useful for plotting.

**stat_ut.py**: Contains statistical functions which are called in src/multiple_regression.

**utilities.py**: Useful functions that don't involve plotting, called from various locations in the code.
