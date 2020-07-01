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


def permutationTestWorker(set_1, set_2, numRuns, workerId, outArray):
    print(f"Worker {workerId} has started.")
    prog = numRuns / 10

    for i in range(0, numRuns):

        if (i % prog == 0):
            print(f"worker {workerId} is done with {int(10*(i/prog))}% of its work")

        perm_1, perm_2 = shuffleLabels(set_1, set_2)
        outArray[workerId*numRuns + i] = np.mean(perm_1) - np.mean(perm_2)

    print(f"worker {workerId} has finished.")

def permutationTest(set_1, set_2, numRuns, numWorkers=8):
    if (numRuns % numWorkers):
        print("ERR: Number of runs not divisible by number of workers")
        return None

    workPerProcess = int(numRuns / numWorkers)
    processes = []
    outArray = mp.Array('f', numRuns, lock=False)

    for i in range(0, numWorkers):
        p = mp.Process(target = permutationTestWorker, args =(set_1, set_2, workPerProcess, i,  outArray))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    res = list(outArray)

    return res


#Does 2tailed
def getPFromPerm(vals, set_1, set_2, numSimulations):

    actualMeanDiff = np.mean(set_1) - np.mean(set_2)
    asExtremeAs = 0

    for val in vals:
        if abs(val) >= abs(actualMeanDiff):
            asExtremeAs += 1

    return (asExtremeAs + 1)/numSimulations
