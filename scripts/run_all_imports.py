#!/bin/bash

# @author Morgan Rivers
# @date 1-25-23

# This script creates all the imported data in the data/processed folder
# see https://stackoverflow.com/questions/64016426/how-to-run-multiple-python-scripts-using-single-python-py-script
import glob, os
from pathlib import Path

# assign the path with all the import scripts


os.chdir("..")  # locate ourselves in the src directory
os.chdir("src")  # locate ourselves in the src directory
os.chdir("import")  # locate ourselves in the import directory

for script in [
    "import_aez.py",
    "import_cell_area_data.py",
    "import_continents.py",
    "import_crop_area_data.py",
    "import_fertilizer.py",
    "import_manure_fertilizer.py",
    "import_pesticide_application_bycrop.py",
    "import_spam_yield_data.py",
    "import_tillage.py",
    "import_irrigation_total.py",
    "import_irrigation_reliant_and_upsample.py",
]:
    # if this causes issues, try changing it to "python". If that doesn't work,
    # perhaps you don't have python3 installed.
    print("")
    print("")
    print("Running the import script:")
    print(script)
    print("")
    os.system("python3 " + script)
