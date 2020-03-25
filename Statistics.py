import pickle
import inspect
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import errno

from ConsoleMessages import ConsoleMessages as cm


MAX_NUM_TO_FILE = 10_000

DATE = datetime.now().strftime('%m-%d_%H-%M')
DATA_ROOT = "/media/max/88B5-59E2/data"
SAVE_IN = DATA_ROOT + "/" + DATE


class ClassCollector():

    def __init__(self, run, owner, savePath, loading = False):
        self.owner = owner
        self.loading = loading

        self.folderPath = f"{savePath}/{self.owner}"

        if not self.loading:
            try:
                os.mkdir(self.folderPath)
            except OSError:
                print(f"{cm.WARNING} Failed to create directory for data of class {self.owner}. Data is lost..{cm.NORMAL}")

        self.statistics = {}

    def addStatistic(self, name, title):

        if name not in self.statistics:
            self.statistics[name] = {"title" : title, "data" : [], "files" : 0}

        statPath = self.folderPath + "/" + name

        if not self.loading:
            try:
                os.mkdir(statPath)
            except OSError:
                print(f"{cm.WARNING} Failed to create directory for statistic {name} in class {self.owner}. Data is lost...{cm.NORMAL}")

        with open(statPath + "/infoFile", "wb") as f:
            pickle.dump(title, f)

    def updateStatistic(self, name, value):

        if name in self.statistics:
            self.statistics[name]["data"].append(value)

            if (len(self.statistics[name]["data"]) >= MAX_NUM_TO_FILE):
                self.saveStatistic(name)
        else:
            print(f"{cm.WARNING} attempted update of unknown statistic" +
            f"{name} by {self.owner}{cm.NORMAL}")

    def saveStatistic(self, name):
        info = self.statistics[name]

        if (len(info["data"]) == 0):
            return

        fileName = self.folderPath + "/" + name + "/d" + str(info["files"])

        with open(fileName, "wb") as f:
            pickle.dump(info["data"], f)

        info["data"] = []
        info["files"] += 1

    def save(self):
        for key, value in self.statistics.items():
            self.saveStatistic(key)

    def _unpack(self, dirName):

        dirPath = f"{self.folderPath}/{dirName}"

        self.statistics[dirName] = {}
        self.statistics[dirName]["data"] = []

        with open(dirPath + "/" + "infoFile", "rb") as f:
            self.statistics[dirName]["title"] = pickle.load(f)
        for file in os.listdir(dirPath):
            if "infoFile" not in file:
                with open(dirPath + "/" + file, "rb") as f:
                    self.statistics[dirName]["data"].extend(pickle.load(f))

    def load(self):
        for folder in os.listdir(self.folderPath):
            self._unpack(folder)

    def summarize(self):
        print(f"Data stored: {len(self.statistics)}\n")

        for key, value in self.statistics.items():
            print(f"statistic: {value['title']}")
            print(f"name: {key}")
            print(f"values recorded: {len(value['data'])}")
            print("---\n")



class RunCollector():

    def __init__(self, run, savePath, loading = False):
        self.runPath = f"{savePath}/run_{run}"
        self.run = run

        self.collectors = {}

        if not loading:

            try:
                os.mkdir(self.runPath)
            except OSError as e:
                print(f"{cm.WARNING} Failed to create directory for data of run {run}. Data is lost...{cm.NORMAL}")
                print(f"{cm.WARNING} error:\t{e}{cm.NORMAL}")
                print(f"{cm.INFO}path:\t{self.runPath}{cm.NORMAL}")


    def getClassCollector(self):
        calledBy = inspect.stack()[2][0].f_locals["self"].__class__.__name__

        if calledBy not in self.collectors:
            self.collectors[calledBy] = ClassCollector(self.run, calledBy, self.runPath)

        return self.collectors[calledBy]

    def save(self):
        for key, value in self.collectors.items():
            value.save()

    def load(self):
        for cc in os.listdir(self.runPath):
            self.collectors[cc] = ClassCollector(self.run, cc, self.runPath, True)
            self.collectors[cc].load()

    def summarize(self):
        print(f"{cm.BACKED_C}-----------{cm.NORMAL}")
        print(f"{cm.BACKED_C}   run {self.run}   {cm.NORMAL}")
        print(f"{cm.BACKED_C}-----------{cm.NORMAL}")
        for key, value in self.collectors.items():
            print(f"{cm.BACKED_P}###Data found for: {key}{cm.NORMAL}\n")
            value.summarize()






#   This class is concerned with storing data from all over the program
#   It is a singleton class, so that all data gets collected at one point
#
#   It also has functionality for plotting the data collected and providing info on it

class StatCollector():

    __instance = None

    loadedDir = None

    currentRun = None

    @staticmethod
    def getInstance(loading = False):
        if StatCollector.__instance == None:
            StatCollector(loading)
        return StatCollector.__instance

    def __init__(self, loading):
        global SAVE_IN

        if StatCollector.__instance != None:
            raise Exception("This class is a singleton - access with getInstance() instead")
        else:
            StatCollector.__instance = self

            if not loading:

                #If another directory has been created in the same minute, then we modify the SAVE_IN
                #until a free one has been found
                tempPath = SAVE_IN
                dirCount = 1

                while (os.path.exists(tempPath)):
                    tempPath = SAVE_IN + "#" + str(dirCount)
                    dirCount += 1

                SAVE_IN = tempPath

                try:
                    os.mkdir(SAVE_IN)
                except OSError:
                    print(f"{cm.WARNING} Failed to create root directory for data. Data is lost..{cm.NORMAL}")

                try:
                    os.mkdir(SAVE_IN + "/rawData")
                except OSError:
                    print(f"{cm.WARNING} Failed to create rawData directory for data. Data is lost..{cm.NORMAL}")

            self.runs = []

    def startRun(self):
        if self.currentRun == None:
            self.currentRun = 0
        else:
            self.save()
            self.currentRun += 1

        self.runs.append(RunCollector(self.currentRun, f"{SAVE_IN}/rawData"))

    def save(self):
        self.runs[self.currentRun].save()

    def getClassCollector(self):
        return self.runs[self.currentRun].getClassCollector()


    def load(self, dirName):
        if not os.path.exists(dirName):
            print(f"{cm.WARNING} Could not find folder {dirName}, aborting loading...{cm.NORMAL}")

        self.loadedDir = dirName

        for run in os.listdir(dirName):
            runFolder = dirName + "/" + run
            loadedRun = RunCollector(len(self.runs), dirName, True)
            self.runs.append(loadedRun)
            loadedRun.load()



    def loadLatest(self, dirName = DATA_ROOT):

        lastDir = None
        lastModification = 0

        for folder in os.listdir(dirName):
            lmTemp = os.path.getmtime(f"{dirName}/{folder}")

            if (lmTemp > lastModification):
                lastModification = lmTemp
                lastDir = folder

        if lastDir is not None:
            self.load(dirName + "/" + lastDir + "/rawData")
        else:
            print(f"{cm.WARNING}No folders found in directory {dirName} to open!{cm.NORMAL}")

    def listFolders(self, dirName = DATA_ROOT):

        for folder in os.listdir(dirName):
            print(f"{cm.INFO}{dirName}/{folder}{cm.NORMAL}")

    def summarize(self):
        print(  f"{cm.NORMAL}\n\n"                  +
                "----------------------------\n"    +
                " Overview of data collected \n"    +
                "----------------------------\n"
        )

        print(f"{cm.INFO}Loaded from folder:\t{self.loadedDir}{cm.NORMAL}")
        print(f"Runs:\t{len(self.runs)}\n\n")

        for run in self.runs:
            run.summarize()

    #TODO
    #
    #   def plotRun()
    #
    #   def plotCombined()
    #
    #
    #
    #   def plotCombinedClass()
    #
    #   def plotCombinedStatistic()
    #
    #
    #
    #
    #
    #
    #


    #TODO make it so that models save in this folder by having the logicModule
    #use this function to get the right directory
    def getCurrentSaveFolder(self):
        return SAVE_IN
