#!/bin/bash

# @author Morgan Rivers
# @date 1-25-23

# This script runs the full loss of industry analysis in the src/multiple_regression folder
# see https://stackoverflow.com/questions/64016426/how-to-run-multiple-python-scripts-using-single-python-py-script
import glob, os
from pathlib import Path

# assign the path with all the import scripts


os.chdir("..")  # locate ourselves in the src directory
os.chdir("src")  # locate ourselves in the src directory
os.chdir("multiple_regression")  # locate ourselves in the import directory


for script in [
    "1_import_data.py",
    "2_data_preprocessing.py",
    "3_LoI_scenario_data.py",
    "4_GLM_analysis_highres.py",
]:
    # if this causes issues, try changing it to "python". If that doesn't work,
    # perhaps you don't have python3 installed.
    print("")
    print("")
    print("Running the analysis script:")
    print(script)
    print("")
    os.system("python3 " + script)
