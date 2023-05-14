🌾📈 `LosingIndustryCropYields`
==============================

Code and notebooks for calculating crop yield in a loss of 
industry scenario. Used in ALLFED research projects.

Code can be run either in the interactive Colab environment (in which case
you don't need to clone this repo locally), or locally using Python. 
Setup instructions for both are below.

# Setup

## Dependencies: Setup in local dev environment 

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

Contact: reach us at morgan@allfed.info 
