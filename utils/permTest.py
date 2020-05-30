import numpy as np
import random
import multiprocessing as mp


#Assumes that len(in_1) == len(in_2)
def shuffleLabels(in_1, in_2):
    if not (len(in_1) == len(in_2)):
        print("Error: shuffleLabels called with 2 sets of unequal length")
        exit(1)

    combinedData = in_1 + in_2
    random.shuffle(combinedData)
    return combinedData[:len(in_1)], combinedData[len(in_2):]


def buildDistributionWorker(set_1, set_2, numRuns, workerId, outArray):
    print(f"Worker {workerId} has started.")
    prog = numRuns / 10

    for i in range(0, numRuns):

        if (i % prog == 0):
            print(f"worker {workerId} is done with {int(10*(i/prog))}% of its work")

        perm_1, perm_2 = shuffleLabels(set_1, set_2)
        outArray[workerId*numRuns + i] = np.mean(perm_1) - np.mean(perm_2)

    print(f"worker {workerId} has finished.")

def buildDistribution(set_1, set_2, numRuns, numBins, numWorkers=8):

    #res = []

    #for i in range(0, numRuns):

    #    if (i % (numRuns/100) == 0):
    #        print(f"Run: {i} / {numRuns}")
    #
    #    perm_1, perm_2 = shuffleLabels(set_1, set_2)
    #    res.append(np.mean(perm_1) - np.mean(perm_2))

    #Parallel approach of above code

    if (numRuns % numWorkers):
        print("ERR: Number of runs not divisible by number of workers")
        return None, None

    workPerProcess = int(numRuns / numWorkers)
    processes = []
    outArray = mp.Array('f', numRuns, lock=False)

    for i in range(0, numWorkers):
        p = mp.Process(target = buildDistributionWorker, args =(set_1, set_2, workPerProcess, i,  outArray))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    res = list(outArray)


    #bin it
    #maxVal(k) ;)
    maxVal = np.max(res)
    minVal = np.min(res)
    print(f"max: {maxVal}, min: {minVal}")
    meanRange = abs(maxVal - minVal)
    stepSize = meanRange / numBins

    hist = [0] * numBins

    #frac = 1 / len(res)

    #WAIT! Doesn't this cause rounding errors?
    for val in res:
        index = int(val / stepSize)
        hist[index] += 1 #frac

    for i in range(0, len(hist)):
        hist[i] /= numRuns

    binVals = []

    for i in range(0, numBins):
        binVals.append(i*stepSize + minVal)

    return hist, binVals

'''
        binvals contain the floor for each bin
        a val belong to a bin if it is equal to or greater than the floor, and
        lower than the ceil (next)

        Special case: As binvals contain floors, then the last ceil needs to be
        manually computed

        There is also the case that the value is contained in no bin whatsoever
        In this case, we return -1
'''

def findBin(binVals, val):
    #for final ceiling
    binStep = abs(binVals[0] - binVals[1])

    if (val < binVals[0]):
        return -1

    for i in range(0, len(binVals)-1):
        if (val >= binVals[i] and val < binVals[i+1]):
            break
        #Custom ceil
        if i+1 == len(binVals)-1:
            if (val >= binVals[i] and val < binVals[i] + binStep):
                break
            else:
                return -1

    return i





#Does 2tailed
def getPFromDist(distribution, binVals, set_1, set_2, numSimulations):

    actualMeanDiff = np.mean(set_1) - np.mean(set_2)

    if actualMeanDiff >= 0:
        lowDif = -actualMeanDiff
        highDif = actualMeanDiff
    else:
        lowDif = actualMeanDiff
        highDif = -actualMeanDiff

    lowBin = findBin(binVals, lowDif)
    highBin = findBin(binVals, highDif)

    asExtreme = 0

    if not (lowBin == -1):
        for i in range(lowBin, -1, -1):
            asExtreme += distribution[i]
    if not (highBin == -1):
        for i in range(highBin, len(distribution)):
            asExtreme += distribution[i]

    return asExtreme + (1/numSimulations)
