import pickle
import inspect
import numpy as np
import matplotlib.pyplot as plt

from ConsoleMessages import ConsoleMessages as cm

class StatCollector():

    __instance = None

    @staticmethod
    def getInstance():
        if StatCollector.__instance == None:
            StatCollector()
        return StatCollector.__instance

    def __init__(self):
        if StatCollector.__instance != None:
            raise Exception("This class is a singleton - access with getInstance() instead")
        else:
            StatCollector.__instance = self
            self.statistics = {}


    def addStatistic(self, name, title):
        calledBy = inspect.stack()[1][0].f_locals["self"].__class__.__name__

        if calledBy not in self.statistics:
            self.statistics[calledBy] = {}

        if name not in self.statistics[calledBy]:
            self.statistics[calledBy][name] = {"title": title, "data" : []}


    def updateStatistic(self, name, value):
        calledBy = inspect.stack()[1][0].f_locals["self"].__class__.__name__

        if calledBy in self.statistics and name in self.statistics[calledBy]:
            self.statistics[calledBy][name]["data"].append(value)
        else:
            print(f"{cm.WARNING} attempted update of unknown statistic" +
            f"{name} by {calledBy}")


    def summarize(self):

        print(  "\n\n"                              +
                "----------------------------\n"    +
                " Overview of data collected \n"    +
                "----------------------------\n"
        )

        for key, value in self.statistics.items():

            print(f"Statistics tracked by class: {key}\n")

            for k2, v2 in value.items():
                print(f"statistic: {v2['title']}")
                print(f"values recorded: {len(v2['data'])}")
                print("-----\n")
            print("######\n")

    def saveData(self, fileName):
        with open(fileName, "wb") as f:
            pickle.dump(self.statistics, f)


    def loadData(self, fileName):
        with open(fileName, "rb") as f:
            self.statistics = pickle.load(f)

    def plotStatistics(self, averageOver = 1):

        plots = 0

        for className, classData in self.statistics.items():

            averagedData = {}
            localAverageOver = averageOver

            for statistic, values in classData.items():
                avg = []

                title = values["title"]
                data = values["data"]

                if (len(data) < localAverageOver):
                    print(f"{cm.WARNING} averageOver is larger than the data averaged over," +
                    f"defaulting it to 1. Plot titles will be incorrect as a result.")
                    localAverageOver = 1
                elif (len(data) % localAverageOver):
                    print(f"{cm.WARNING} averageOver {localAverageOver} does not divide data of length " +
                    f"{len(data)} wholly, plots may be incorrect as a result.]")

                chunks = int(len(data) / localAverageOver)


                for i in range(0, chunks):
                    avg.append(np.average(data[i*localAverageOver:(i+1)*localAverageOver]))

                averagedData[title] = avg

            # At this point all data of 1 class is averaged and
            # stored with as key the title that needs to be used

            plt.figure(plots)
            plotCtr = 1

            for title, data in averagedData.items():

                # We allow at most 4 subplots per plot
                if (plotCtr > 4):
                    plotCtr = 1
                    plots += 1
                    plt.figure(plots)

                plt.subplot(2, 2, plotCtr)
                plt.plot(data)

                plt.title(title + f", averaged over {averageOver} runs")
                plotCtr += 1
            plots += 1
        plt.show()
