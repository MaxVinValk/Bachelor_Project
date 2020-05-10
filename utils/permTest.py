import numpy as np
import random


#Assumes that len(in_1) == len(in_2)
def shuffleLabels(in_1, in_2):
    if not (len(in_1) == len(in_2)):
        print("Error: shuffleLabels called with 2 sets of unequal length")
        exit(1)

    allData = random.shuffle(in_1 + in_2)
    return allData[:len(in_1)], allData[len(in_2):]



def buildDistribution(set_1, set_2, numRuns, numBins):

    res = []

    for i in range(0, numRuns):
        perm_1, perm_2 = shuffleLabels(set_1, set_2)
        res.append(np.mean(perm_1) - np.mean(perm_2))

    #bin it
    #maxVal(k) ;)
    maxVal = np.max(res)
    minVal = np.min(res)
    meanRange = abs(maxVal - minVal)
    stepSize = meanRange / numBins

    hist = [0] * numBins

    frac = 1 / len(res)

    for val in res:
        index = int(val / stepSize)
        hist[index] += frac

    binVals = []

    for i in range(0, len(numBins)):
        binVals.append(i*numBins)

    return hist, binVals


def getPFromDist(distribution, binVals, set_1, set_2):

    actualMeanDiff = np.mean(set_1) - np.mean(set_2)


    #Get the right bin
    bin = binVals[0]

    for i in range(0, len(binVals)):
        if actualMeanDiff < binVals[i]:
            break
        bin = binVals[i]

    #The bin gives us p:
    return binVals[bin]


# TODO: Add ABL loading, own loading
dataABL = []
dataQNN = []

#TODO: separate in different categories of interest:
#       Final accuracy, time of reaching 50% accuracy, 75% accuracy

#TODO: function calls to get p values
