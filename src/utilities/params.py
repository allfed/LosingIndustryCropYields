import os

import git
from pathlib import Path

repo_root = git.Repo(".", search_parent_directories=True).working_dir

paramsfilename = str(Path(repo_root) / "src" / "misc" / "Params.ods")


def importIfNotAlready():
    global paramsinitialized
    try:
        paramsinitialized
    except NameError:  # if the params haven't been initialized yet
        paramsinitialized = True
        importAll()
    else:
        return ()


def deleteGlobals():
    del spamCropYieldDataLoc
    del cropAreaDataLoc
    del pesticidesDataLoc
    del tillageDataLoc
    del aezDataLoc
    del fertilizerDataLoc
    del manureFertilizerDataLoc
    del irrigationDataLoc
    del livestockDataLoc
    del aquastatIrrigationDataLoc
    del cropYieldDataLoc
    del geopandasDataDir
    del figuresDir
    del growAreaDataLoc
    # del tempCSVloc
    # del windCSVloc
    # del temphumsunrainCSVloc
    del outputAsciiDir
    del inputDataDir
    del cropDataDir
    del modelDataDir
    del LoIDataDir
    del figuresDir
    del statisticsDir
    del inputAsciiDir
    del latdiff
    del londiff
    del growAreaBins
    del allMonths
    del plotTemps
    del plotRain
    del plotSun
    del plotYield
    del plotGrowArea
    del plotTempCoeffs
    del plotRainCoeffs
    del saveTempCSV
    del estimateNutrition
    del estimateYield
    del allCrops
    del rain_mps_to_mm
    """
    del Tbase
    del Tfp
    del Topt1
    del Topt2
    del RpeakCoeff  
    del RlowCoeff   
    del RhighCoeff  
    del Rlow    
    del Rpeak   
    del Rhigh
    del growDuration
    del idealGrowth
    del kCalperkg
    del fracProtein
    del fracFat
    del fracCarbs
"""


def importAll():
    importDirectories()
    importModelParams()
    # importYieldTemp()
    # importYieldRain()
    # importGrowingSeason()
    # importNutrition()


def importDirectories():
    from sys import platform
    from pyexcel_ods3 import get_data

    global spamCropYieldDataLoc
    global cropAreaDataLoc
    global pesticidesDataLoc
    global tillageDataLoc
    global aezDataLoc
    global fertilizerDataLoc
    global manureFertilizerDataLoc
    global irrigationDataLoc
    global livestockDataLoc
    global aquastatIrrigationDataLoc
    global cropYieldDataLoc
    global continentDataLoc
    global growAreaDataLoc
    global inputDataDir
    global cropDataDir
    global modelDataDir
    global LoIDataDir
    global figuresDir
    global statisticsDir
    global inputAsciiDir
    global geopandasDataDir
    # global tempCSVloc
    # global windCSVloc
    # global temphumsunrainCSVloc
    global outputAsciiDir

    data = get_data(paramsfilename)
    paramdata = data["Directory"]

    for coltitleindex in range(0, len(paramdata[1])):
        coltitle = paramdata[1][coltitleindex]
        if coltitle == "cropYieldDataLoc":
            cropYieldDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "spamCropYieldDataLoc":
            spamCropYieldDataLoc = (
                str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
            )
        if coltitle == "cropAreaDataLoc":
            cropAreaDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex])
        if coltitle == "pesticidesDataLoc":
            pesticidesDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "tillageDataLoc":
            tillageDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex])
        if coltitle == "aezDataLoc":
            aezDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "fertilizerDataLoc":
            fertilizerDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "manureFertilizerDataLoc":
            manureFertilizerDataLoc = (
                str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
            )
        if coltitle == "irrigationDataLoc":
            irrigationDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "livestockDataLoc":
            livestockDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "aquastatIrrigationDataLoc":
            aquastatIrrigationDataLoc = str(
                Path(repo_root) / paramdata[2][coltitleindex]
            )
        if coltitle == "continentDataLoc":
            continentDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "growAreaDataLoc":
            growAreaDataLoc = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "geopandasDataDir":
            geopandasDataDir = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "inputDataDir":
            inputDataDir = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "cropDataDir":
            cropDataDir = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "modelDataDir":
            modelDataDir = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "LoIDataDir":
            LoIDataDir = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "figuresDir":
            figuresDir = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        if coltitle == "statisticsDir":
            statisticsDir = str(Path(repo_root) / paramdata[2][coltitleindex]) + "/"
        # if(coltitle == 'tempCSVloc'):
        # tempCSVloc=dir_path+'/../'+paramdata[2][coltitleindex]
        # if(coltitle == 'temphumsunrainCSVloc'):
        # temphumsunrainCSVloc=dir_path+'/../'+paramdata[2][coltitleindex]
        # if(coltitle == 'windCSVloc'):
        # windCSVloc=dir_path+'/../'+paramdata[2][coltitleindex]
        if coltitle == "inputAsciiDir":
            inputAsciiDir = str(Path(repo_root) / paramdata[2][coltitleindex])
        if coltitle == "outputAsciiDir":
            outputAsciiDir = str(Path(repo_root) / paramdata[2][coltitleindex])


def importModelParams():
    from pyexcel_ods3 import get_data

    global latdiff
    global londiff
    global growAreaBins
    global allMonths
    global plotTemps
    global plotRain
    global plotSun
    global plotYield
    global plotGrowArea
    global plotTempCoeffs
    global plotRainCoeffs
    global saveTempCSV
    global estimateNutrition
    global estimateYield
    global allCrops
    global rain_mps_to_mm

    data = get_data(paramsfilename)
    paramdata = data["ModelParams"]

    for coltitleindex in range(0, len(paramdata[1])):
        coltitle = paramdata[1][coltitleindex]
        if coltitle == "latdiff":
            latdiff = paramdata[2][coltitleindex]
        if coltitle == "londiff":
            londiff = paramdata[2][coltitleindex]
        if coltitle == "growAreaBins":
            growAreaBins = paramdata[2][coltitleindex]
        if coltitle == "rain_mps_to_mm":
            rain_mps_to_mm = paramdata[2][coltitleindex]
        if coltitle == "allMonths":
            am = []
            for i in range(0, len(paramdata)):
                if i < 2:
                    continue
                m = paramdata[i]
                if not m:
                    break
                am.append(m[coltitleindex])
            allMonths = am
        if coltitle == "allCrops":
            ac = []
            for i in range(0, len(paramdata)):
                if i < 2:
                    continue
                c = paramdata[i]
                if not c:
                    break
                if len(c) - 1 < coltitleindex:
                    break
                ac.append(c[coltitleindex])
            allCrops = ac
        if coltitle == "plotTemps":
            plotTemps = (
                paramdata[2][coltitleindex] == "TRUE"
                or paramdata[2][coltitleindex] == True
            )
        if coltitle == "plotRain":
            plotRain = (
                paramdata[2][coltitleindex] == "TRUE"
                or paramdata[2][coltitleindex] == True
            )
        if coltitle == "plotSun":
            plotSun = (
                paramdata[2][coltitleindex] == "TRUE"
                or paramdata[2][coltitleindex] == True
            )
        if coltitle == "plotYield":
            plotYield = (
                paramdata[2][coltitleindex] == "TRUE"
                or paramdata[2][coltitleindex] == True
            )
        if coltitle == "plotGrowArea":
            plotGrowArea = (
                paramdata[2][coltitleindex] == "TRUE"
                or paramdata[2][coltitleindex] == True
            )
        # if(coltitle == 'plotTempCoeff'):
        # plotTempCoeff = (paramdata[2][coltitleindex]=='TRUE' or paramdata[2][coltitleindex]==True)
        # if(coltitle == 'plotRainCoeff'):
        # plotRainCoeff = (paramdata[2][coltitleindex]=='TRUE' or paramdata[2][coltitleindex]==True)
        # if(coltitle == 'saveTempCSV'):
        # saveTempCSV = (paramdata[2][coltitleindex]=='TRUE' or paramdata[2][coltitleindex]==True)
        if coltitle == "estimateYield":
            estimateYield = (
                paramdata[2][coltitleindex] == "TRUE"
                or paramdata[2][coltitleindex] == True
            )
        if coltitle == "estimateNutrition":
            estimateNutrition = (
                paramdata[2][coltitleindex] == "TRUE"
                or paramdata[2][coltitleindex] == True
            )


"""
def importYieldTemp():

    from pyexcel_ods3 import get_data

    global Tbase
    global Tfp
    global Topt1
    global Topt2

    data = get_data(paramsfilename)
    paramdata = data['YieldTempCoeff']

    Tbase={}
    Tfp={}
    Topt1={}
    Topt2={}

    for cropindex in range(2,len(paramdata)):

        croprow=paramdata[cropindex]
        if(not croprow):
            break

        crop=croprow[0]
        for coltitleindex in range(1,len(paramdata[1])):
            coltitle=paramdata[1][coltitleindex]
            if(coltitle == 'Tbase'):
                Tbase[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'Tfp'):
                Tfp[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'Topt1'):
                Topt1[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'Topt2'):
                Topt2[crop]=paramdata[cropindex][coltitleindex]


def importYieldRain():
    from pyexcel_ods3 import get_data
    global RpeakCoeff   
    global RlowCoeff    
    global RhighCoeff   
    global Rlow 
    global Rpeak    
    global Rhigh

    data = get_data(paramsfilename)
    paramdata = data['YieldRainCoeff']

    RpeakCoeff={}
    RlowCoeff={}
    RhighCoeff={}
    Rlow={}
    Rpeak={}
    Rhigh={}


    for cropindex in range(2,len(paramdata)):

        croprow=paramdata[cropindex]
        if(not croprow):
            break

        crop=croprow[0]
        for coltitleindex in range(1,len(paramdata[1])):
            coltitle=paramdata[1][coltitleindex]
            if(coltitle == 'RpeakCoeff'):
                RpeakCoeff[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'RlowCoeff'):
                RlowCoeff[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'RhighCoeff'):
                RhighCoeff[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'Rlow'):
                Rlow[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'Rpeak'):
                Rpeak[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'Rhigh'):
                Rhigh[crop]=paramdata[cropindex][coltitleindex]


def importGrowingSeason():
    from pyexcel_ods3 import get_data

    global growDuration
    global idealGrowth

    data = get_data(paramsfilename)
    paramdata = data['GrowingSeason']

    growDuration = {}
    idealGrowth = {}

    for cropindex in range(2,len(paramdata)):

        croprow=paramdata[cropindex]
        if(not croprow):
            break

        crop=croprow[0]
        for coltitleindex in range(1,len(paramdata[1])):
            coltitle=paramdata[1][coltitleindex]
            if(coltitle == 'growDuration'):
                growDuration[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'idealGrowth'):
                idealGrowth[crop]=paramdata[cropindex][coltitleindex]


def importNutrition():
    from pyexcel_ods3 import get_data

    global kCalperkg
    global fracProtein
    global fracFat
    global fracCarbs

    data = get_data(paramsfilename)
    paramdata = data['Nutrition']

    kCalperkg = {}
    fracProtein = {}
    fracFat = {}
    fracCarbs = {}

    for cropindex in range(2,len(paramdata)):

        croprow=paramdata[cropindex]
        if(not croprow):
            break

        crop=croprow[0]
        for coltitleindex in range(1,len(paramdata[1])):
            coltitle=paramdata[1][coltitleindex]
            if(coltitle == 'kCalperkg'):
                kCalperkg[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'fracProtein'):
                fracProtein[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'fracFat'):
                fracFat[crop]=paramdata[cropindex][coltitleindex]
            if(coltitle == 'fracCarbs'):
                fracCarbs[crop]=paramdata[cropindex][coltitleindex]
"""
