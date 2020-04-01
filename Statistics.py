import pickle
import inspect
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import errno
from copy import deepcopy

from ConsoleMessages import ConsoleMessages as cm

#TODO: Move loading to settings!
#TODO: Move other components to settings, too


class StatSettings():

    outputData = True

    def __init__(self):
        pass

    def enableDataOutput(self):
        self.outputData = True

    def disableDataOutput(self):
        self.outputData = False


class ClassCollector():

    #TODO: Move to settings
    MAX_NUM_TO_FILE = 10_000

    settings = None

    def __init__(self, run, owner, savePath, settings, loading = False):
        self.owner = owner
        self.loading = loading
        self.settings = settings

        self.folderPath = f"{savePath}/{self.owner}"

        self.statistics = {}

        if self.loading or not self.settings.outputData:
            return

        try:
            os.mkdir(self.folderPath)
        except OSError:
            print(f"{cm.WARNING} Failed to create directory for data of class {self.owner}. Data is lost..{cm.NORMAL}")



    def addStatistic(self, name, title):

        if name not in self.statistics:
            self.statistics[name] = {"title" : title, "data" : [], "files" : 0}

        statPath = self.folderPath + "/" + name

        if self.loading or not self.settings.outputData:
            return

        try:
            os.mkdir(statPath)
        except OSError:
            print(f"{cm.WARNING} Failed to create directory for statistic {name} in class {self.owner}. Data is lost...{cm.NORMAL}")

        with open(statPath + "/infoFile", "wb") as f:
            pickle.dump(title, f)

    def updateStatistic(self, name, value):

        if name in self.statistics:
            self.statistics[name]["data"].append(value)

            if (len(self.statistics[name]["data"]) >= self.MAX_NUM_TO_FILE):
                self.saveStatistic(name)
        else:
            print(f"{cm.WARNING} attempted update of unknown statistic" +
            f"{name} by {self.owner}{cm.NORMAL}")

    def saveStatistic(self, name):

        if not self.settings.outputData:
            return

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
        for file in sorted(os.listdir(dirPath)):
            if "infoFile" not in file:
                with open(dirPath + "/" + file, "rb") as f:
                    self.statistics[dirName]["data"].extend(pickle.load(f))

    def load(self):
        for folder in sorted(os.listdir(self.folderPath)):
            self._unpack(folder)

    def summarize(self):
        print(f"Data stored: {len(self.statistics)}\n")

        for key, value in self.statistics.items():
            print(f"statistic: {value['title']}")
            print(f"name: {key}")
            print(f"values recorded: {len(value['data'])}")
            print("---\n")

    def sumWithOtherCC(self, cc):

        for key, value in self.statistics.items():
            if key in cc.statistics:
                dataOwn = value["data"]
                dataOther = cc.statistics[key]["data"]

                if (len(dataOwn) != len(dataOther)):
                    print(f"{cm.INFO}attempted sum of two class collectors who do not hold same" +
                    f"amount of values {cm.NORMAL}")
                else:
                    for i in range(0, len(dataOwn)):
                        dataOwn[i] += dataOther[i]

    def divideAllData(self, div):
        for key, value in self.statistics.items():
            for i in range(0, len(value["data"])):
                value["data"][i] /= div



    def _averageStatOver(self, name, averageOver):

        localAverageOver = averageOver
        avg = []
        data = self.statistics[name]["data"]

        if (len(data) < localAverageOver):
            print(f"{cm.WARNING} averageOver is larger than the data averaged over," +
            f"defaulting it to 1. Plot titles will be incorrect as a result.{cm.NORMAL}")
            localAverageOver = 1
        elif (len(data) % localAverageOver):
            print(f"{cm.WARNING} averageOver {localAverageOver} does not divide data of length " +
            f"{len(data)} wholly, plots may be incorrect as a result.{cm.NORMAL}")

        chunks = int(len(data) / localAverageOver)

        for i in range(0, chunks):
            avg.append(np.average(data[i*localAverageOver:(i+1)*localAverageOver]))

        return avg


    def readyPlot(self, averageOver, plots, shape = [2, 2], plotAll = True, *toPlot):

        avgData = {}
        maxSubPlots = shape[0] * shape[1]

        if (plotAll):
            for key, value in self.statistics.items():
                avgData[value["title"]] = self._averageStatOver(key, averageOver)
        else:
            for stat in toPlot:
                print(stat)
                if stat in self.statistics:
                    value = self.statistics[stat]
                    avgData[value["title"]] = self._averageStatOver(stat, averageOver)
                else:
                    print(f"{cm.INFO}Could not find data for statistic {stat} in class" +
                    f"{self.owner}{cm.NORMAL}")


        plotCtr = 1

        for title, data in avgData.items():

            if (plotCtr == 1):
                plt.figure(plots)

            plt.subplot(shape[0], shape[1], plotCtr)
            plt.plot(data)

            plt.title(title + f", averaged over {averageOver} runs")
            plotCtr += 1

            if (plotCtr > maxSubPlots):
                plotCtr = 1
                plots += 1



        if (plotCtr == 1):
            return plots
        else:
            return plots + 1


    def getStatistic(self, statisticName):
        if statisticName in self.statistics:
            return self.statistics[statisticName]
        else:
            print(f"{cm.ERROR} Could not find requested statistic {statisticName} in class {self.owner}.{cm.NORMAL}")



class RunCollector():

    outputData = True
    settings = None

    def __init__(self, run, savePath, settings, loading = False):

        self.runPath = f"{savePath}/run_{run}"
        self.run = run
        self.settings = settings

        self.collectors = {}

        if loading or not self.settings.outputData:
            return

        try:
            os.mkdir(self.runPath)
        except OSError as e:
            print(f"{cm.WARNING} Failed to create directory for data of run {run}. Data is lost...{cm.NORMAL}")
            print(f"{cm.WARNING} error:\t{e}{cm.NORMAL}")
            print(f"{cm.INFO}path:\t{self.runPath}{cm.NORMAL}")


    def getClassCollector(self):
        calledBy = inspect.stack()[2][0].f_locals["self"].__class__.__name__

        if calledBy not in self.collectors:
            self.collectors[calledBy] = ClassCollector(self.run, calledBy, self.runPath, self.settings)

        return self.collectors[calledBy]

    def save(self):
        for key, value in self.collectors.items():
            value.save()


    def load(self):
        for cc in sorted(os.listdir(self.runPath)):
            self.collectors[cc] = ClassCollector(self.run, cc, self.runPath, self.settings, True)
            self.collectors[cc].load()

    def summarize(self):

        print(f"{cm.BACKED_C}-----------{cm.NORMAL}")
        print(f"{cm.BACKED_C}   run {self.run}   {cm.NORMAL}")
        print(f"{cm.BACKED_C}-----------{cm.NORMAL}")
        for key, value in self.collectors.items():
            print(f"{cm.BACKED_P}###Data found for: {key}{cm.NORMAL}\n")
            value.summarize()

    def combineRunData(self, run):
        for key, value in self.collectors.items():
            if key in run.collectors:
                value.sumWithOtherCC(run.collectors[key])


    def divideAllData(self, div):
        for key, value in self.collectors.items():
            value.divideAllData(div)

    def getData(self):
        return self.collectors

    def plot(self, averageOver = 1, shape = [2, 2], plotAll = True, *toPlot):
        plots = 0

        for key, value in self.collectors.items():
            plots = value.readyPlot(averageOver, plots, shape, plotAll, *toPlot)

        plt.show()

    def getStatistic(self, className, statisticName):

        if className in self.collectors:
            return self.collectors[className].getStatistic(statisticName)
        else:
            print(f"{cm.WARNING} could not find class collector for {className} when attempting to retrieve statistic {statisticName} {cm.NORMAL}")



#   This class is concerned with storing data from all over the program
#   It is a singleton class, so that all data gets collected at one point
#
#   It also has functionality for plotting the data collected and providing info on it

class StatCollector():

    __instance = None

    loadedDir = None
    currentRun = None

    settings = StatSettings()
    sessionData = {}


    #move these to statSettings
    date = datetime.now().strftime('%m-%d_%H-%M')
    #dataRoot should be constant> Or changeable with a function
    dataRoot = "/home/maxvalk/Documents/Uni/Scriptie/data_out"
    saveIn = dataRoot + "/" + date

    outputData = True

    @staticmethod
    def getInstance(loading = False):
        if StatCollector.__instance == None:
            StatCollector(loading)
        return StatCollector.__instance

    def __init__(self, loading):

        if StatCollector.__instance != None:
            raise Exception("This class is a singleton - access with getInstance() instead")
        else:
            StatCollector.__instance = self

            self.runs = []

    def setDataRoot(self, dataRoot):
        self.dataRoot = dataRoot

    def startSession(self):

         self.date = datetime.now().strftime('%m-%d_%H-%M')
         self.saveIn = self.dataRoot + "/" + self.date

         if not self.settings.outputData:
             return

         #If another directory has been created in the same minute, then we modify the saveIn
         #until a free one has been found
         tempPath = self.saveIn
         dirCount = 1

         while (os.path.exists(tempPath)):
             tempPath = self.saveIn + "#" + str(dirCount)
             dirCount += 1

         self.saveIn = tempPath

         try:
             os.mkdir(self.saveIn)
         except OSError:
             print(f"{cm.WARNING} Failed to create root directory for data. Data is lost..{cm.NORMAL}")

         try:
             os.mkdir(self.saveIn + "/rawData")
         except OSError:
             print(f"{cm.WARNING} Failed to create rawData directory for data. Data is lost..{cm.NORMAL}")


    def startRun(self):
        if self.currentRun == None:
            self.currentRun = 0
        else:
            self.save()
            #No need to keep old runs in memory
            self.runs[self.currentRun] = None

            self.currentRun += 1

        self.runs.append(RunCollector(self.currentRun, f"{self.saveIn}/rawData", self.settings))

    def save(self):
        self.runs[self.currentRun].save()

        if (len(self.sessionData)):
            with open(self.saveIn + "/rawData/infoFile", "wb") as f:
                pickle.dump(self.sessionData, f)
                self.sessionData = {}



    def getClassCollector(self):
        return self.runs[self.currentRun].getClassCollector()

    def getStatistic(self, run, className, statisticName):
        return run.getStatistic(className, statisticName)

    def getStatisticLatest(self, className, statisticName):
        return self.runs[self.currentRun].getStatistic(className, statisticName)


    def load(self, dirName):

        if not os.path.exists(dirName):
            print(f"{cm.WARNING} Could not find folder {dirName}, aborting loading...{cm.NORMAL}")

        self.loadedDir = dirName
        self.currentRun = 0
        self.runs = []

        for run in sorted(os.listdir(dirName)):

            if os.path.isfile(f"{dirName}/{run}"):
                continue

            runFolder = dirName + "/" + run
            loadedRun = RunCollector(len(self.runs), dirName, self.settings, True)
            self.runs.append(loadedRun)
            loadedRun.load()

        if os.path.exists(dirName + "/infoFile"):
            with open(dirName + "/infoFile", "rb") as f:
                self.sessionData = pickle.load(f)
        else:
            print(f"{cm.INFO}Could not find infofile {dirName + '/infoFile'}{cm.NORMAL}")

    def loadLatest(self, dirName = None):

        if dirName is None:
            dirName = self.dataRoot

        lastDir = None
        lastModification = 0

        for folder in os.listdir(dirName):

            if os.path.isfile(f"{dirName}/{folder}"):
                continue

            lmTemp = os.path.getmtime(f"{dirName}/{folder}")

            if (lmTemp > lastModification):
                lastModification = lmTemp
                lastDir = folder

        if lastDir is not None:
            self.load(dirName + "/" + lastDir + "/rawData")
        else:
            print(f"{cm.WARNING}No folders found in directory {dirName} to open!{cm.NORMAL}")


    def addSessionData(self, name, value):

        if name not in self.sessionData:
            self.sessionData[name] = value
        else:
            print(f"{cm.WARNING}Trying to add duplicate session data {name}. {cm.NORMAL}")


    def listFolders(self, dirName = None):

        if dirName is None:
            dirName = self.dataRoot


        for folder in os.listdir(dirName):
            print(f"{cm.INFO}{dirName}/{folder}{cm.NORMAL}")

    def summarize(self, detailed = False):
        print(  f"{cm.NORMAL}\n\n"                  +
                "----------------------------\n"    +
                " Overview of data collected \n"    +
                "----------------------------\n"
        )

        print(f"{cm.INFO}Loaded from folder:\t{self.loadedDir}{cm.NORMAL}")

        print(f"{cm.INFO}Session data collected:{cm.NORMAL}")
        for key, value in self.sessionData.items():
            print(f"{key}\t{value}")

        print("@@@@@@@@@@@@")
        if detailed:
            print("\n")
            for run in self.runs:
                run.summarize()
        else:
            print(f"Number of runs collected: {len(self.runs)}")


    def summarizeRun(self, runIdx):
        self._summarizeRun(self, self.runs[runIdx])

    def _summarizeRun(self, run):
        run.summarize()

    #Combines in 1 run the statistics of all runs
    #by averaging values at the storage index
    def averagedRun(self):
        storageObj = None

        for run in self.runs:
            if storageObj == None:
                storageObj = deepcopy(run)
            else:
                storageObj.combineRunData(run)

        storageObj.divideAllData(len(self.runs))

        return storageObj

    #Allows for plotting a run with the index of the run
    def plotRun(self, runIdx, averageOver = 1, shape = [2, 2], plotAll = True, *toPlot):
        self._plotRun(self.runs[runIdx], averageOver, shape, plotAll, *toPlot)

    def plotAverage(self, averageOver = 1, shape = [2, 2], plotAll = True, *toPlot):
        averagedRun = self.averagedRun()
        self._plotRun(averagedRun, averageOver, shape, plotAll, *toPlot)

    #plots a run with a runcollector as input
    def _plotRun(self, run, averageOver, shape, plotAll, *toPlot):
        run.plot(averageOver, shape, plotAll, *toPlot)

    #TODO make it so that models save in this folder by having the logicModule
    #use this function to get the right directory
    def getCurrentSaveFolder(self):
        return self.saveIn

    def getSettings(self):
        return self.settings
