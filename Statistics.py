import pickle
import inspect
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import errno

from ConsoleMessages import ConsoleMessages as cm

MAX_NUM_TO_FILE = 10_000
SAVE_WHILE_RUNNING = False

DATE = datetime.now().strftime('%m-%d_%H-%M')
SAVE_IN = "/media/max/88B5-59E2/data/" + DATE

#TODO: Save data on the fly

class ClassCollector():

    def __init__(self, owner, load = False):
        self.owner = owner

        self.folderPath = SAVE_IN + "/rawData/" + self.owner

        if (not load):
            try:
                os.mkdir(self.folderPath)
            except OSError:
                print(f"{cm.WARNING} Failed to create directory for data. Data is lost...{cm.NORMAL}")

        self.statistics = {}

    def addStatistic(self, name, title):
        if name not in self.statistics:
            self.statistics[name] = {"title" : title, "data" : [], "files" : 0}

        statPath = self.folderPath + "/" + name

        try:
            os.mkdir(statPath)
        except OSError:
            print(f"{cm.WARNING} Failed to create directory for data. Data is lost...{cm.NORMAL}")

        with open(statPath + "/infoFile", "wb") as f:
            pickle.dump(title, f)

    def updateStatistic(self, name, value):
        if name in self.statistics:
            self.statistics[name]["data"].append(value)
            self.saveStatistic(name)
        else:
            print(f"{cm.WARNING} attempted update of unknown statistic" +
            f"{name} by {calledBy}{cm.NORMAL}")


    def saveStatistic(self, name):
        info = self.statistics[name]

        if (len(info["data"]) >= MAX_NUM_TO_FILE):

            fileName = self.folderPath + "/" + name + "/d" + str(info["files"])

            #write to file
            with open(fileName, "wb") as f:
                pickle.dump(info["data"], f)
            info["data"] = []
            info["files"] += 1


    def getStatistics(self):
        return self.statistics

    def save(self):

        for key, value in self.statistics.items():

            if (SAVE_WHILE_RUNNING):
                self.saveStatistic(key)
            else:
                localDir = self.folderPath + "/" + key

                data = value["data"]
                chunks = int(len(data) / MAX_NUM_TO_FILE)
                fileCtr = 0
                filePrefix = localDir + "/d_"

                for i in range(0, chunks):
                    fileName = filePrefix + str(fileCtr)

                    with open(fileName, "wb") as f:
                        pickle.dump(data[i*MAX_NUM_TO_FILE : (i+1)*MAX_NUM_TO_FILE], f)
                    fileCtr += 1

                rem = len(data) % MAX_NUM_TO_FILE

                if (rem):
                    fileName = filePrefix + str(fileCtr)
                    with open(fileName, "wb") as f:
                        pickle.dump(data[-rem:], f)


    def _unpack(self, dirPath, dirName):

        self.statistics[dirName] = {}
        self.statistics[dirName]["data"] = []

        with open(dirPath + "/" + "infoFile", "rb") as f:
            self.statistics[dirName]["title"] = pickle.load(f)
        for file in os.listdir(dirPath):
            if "infoFile" not in file:

                with open(dirPath + "/" + file, "rb") as f:
                    self.statistics[dirName]["data"].extend(pickle.load(f))


    def load(self, dirName):
        for folder in os.listdir(dirName):
            self._unpack(dirName + "/" + folder, folder)


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



#
#   This class is concerned with storing data from all over the program
#   It is a singleton class, so that all data gets collected at one point
#
#   It also has functionality for plotting the data collected and providing info on it

class StatCollector():

    __instance = None

    @staticmethod
    def getInstance(load = False):
        if StatCollector.__instance == None:
            StatCollector(load)
        return StatCollector.__instance

    def __init__(self, load):
        if StatCollector.__instance != None:
            raise Exception("This class is a singleton - access with getInstance() instead")
        else:
            StatCollector.__instance = self
            self.collectors = {}
            self.statistics = {}

            if (not load):
                try:
                    os.mkdir(SAVE_IN)
                except OSError:
                    print(f"{cm.WARNING} Failed to create root directory for data. Data is lost..{cm.NORMAL}")

                try:
                    os.mkdir(SAVE_IN + "/rawData")
                except OSError:
                    print(f"{cm.WARNING} Failed to create rawData directory for data. Data is lost..{cm.NORMAL}")




    def getClassCollector(self):
        calledBy = inspect.stack()[1][0].f_locals["self"].__class__.__name__

        if calledBy not in self.collectors:
            self.collectors[calledBy] = ClassCollector(calledBy)

        return self.collectors[calledBy]



    def save(self):
        for key, value in self.collectors.items():
            value.save(dirName)


    def load(self, dirName):

        for folder in os.listdir(dirName):
            self.collectors[folder] = ClassCollector(folder, True)
            self.collectors[folder].load(dirName + "/" + folder)


    def summarize(self):

        print(  f"{cm.NORMAL}\n\n"                  +
                "----------------------------\n"    +
                " Overview of data collected \n"    +
                "----------------------------\n"
        )

        for key, value in self.collectors.items():
            print(f"{cm.BACKED_P}###Data found for: {key}{cm.NORMAL}\n")

            for k2, v2 in value.getStatistics().items():
                print(f"{cm.NORMAL}statistic: {v2['title']}{cm.NORMAL}")
                print(f"{cm.NORMAL}name: {k2}{cm.NORMAL}")
                print(f"{cm.NORMAL}values recorded: {len(v2['data'])}{cm.NORMAL}")
                print("-----\n")
            print("######\n")

    def plot(self, averageOver = 1, shape = [2, 2]):
        plots = 0

        for key, value in self.collectors.items():
            plots = value.readyPlot(averageOver, plots, shape)

        plt.show()

    def plotClass(self, className, averageOver = 1, shape = [2, 2]):

        if className in self.collectors:
            plots = self.collectors[className].readyPlot(averageOver, 0, shape)
            plt.show()
        else:
            print(f"{cm.INFO}Could not find data for class {className}{cm.NORMAL}")

    def plotStatistic(self, className, statName, averageOver = 1):

        if className in self.collectors:
            self.collectors[className].readyPlot(averageOver, 0, [1, 1], False, statName)
            plt.show()
        else:
            print(f"{cm.INFO}Could not find data for class {className}{cm.NORMAL}")
